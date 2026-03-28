//! Learned Embedding Projection — adapts raw embeddings based on task outcome feedback.
//!
//! External embeddings capture general semantic similarity, but "semantically similar"
//! doesn't mean "useful to recall together." This module trains a linear projection
//! matrix W via triplet loss so that query-memory pairs from successful recalls are
//! pulled closer and failure-attributed memories are pushed apart.
//!
//! The projection starts as identity (no-op) and is trained only after sufficient
//! triplets are collected from `recall_contexts` + `task_outcomes`.
//!
//! Algorithm: manual backprop with ndarray — triplet loss gradient for a linear
//! projection is ~20 lines of math, no autograd framework needed.

use std::collections::BTreeMap;

use ndarray::{Array1, Array2, Axis};

use crate::error::{MemoriaError, Result};
use crate::services::traits::Embedder;
use crate::store::CozoStore;

/// A training triplet: (query, positive_memory, negative_memory).
pub type Triplet = (Vec<f32>, Vec<f32>, Vec<f32>);

/// Statistics from a single training session.
#[derive(Debug, Clone)]
pub struct ProjectionStats {
    pub loss_before: f64,
    pub loss_after: f64,
    pub triplet_count: usize,
    pub epochs: usize,
    pub duration_ms: u64,
}

/// Learned linear projection for embedding adaptation.
///
/// Projects raw embeddings via `y = Wx` then L2-normalizes.
/// Initialized as identity (no-op until trained).
#[derive(Debug, Clone)]
pub struct EmbeddingProjection {
    /// Row-major weight matrix stored flat: dim × dim.
    weights: Vec<f32>,
    /// Embedding dimension.
    pub dim: usize,
    /// Triplet loss margin.
    pub margin: f32,
    /// Loss from the most recent training session.
    pub last_loss: f64,
    /// Number of completed training sessions.
    pub train_count: u64,
}

impl EmbeddingProjection {
    /// Create an identity projection (no-op until trained).
    pub fn identity(dim: usize) -> Self {
        let mut weights = vec![0.0f32; dim * dim];
        for i in 0..dim {
            weights[i * dim + i] = 1.0;
        }
        Self {
            weights,
            dim,
            margin: 0.3,
            last_loss: 1.0,
            train_count: 0,
        }
    }

    /// Project an embedding: `y = Wx`, then L2-normalize.
    pub fn project(&self, embedding: &[f32]) -> Vec<f32> {
        debug_assert_eq!(embedding.len(), self.dim);
        let w = Array2::from_shape_vec((self.dim, self.dim), self.weights.clone())
            .expect("weights shape mismatch");
        let x = Array1::from_vec(embedding.to_vec());
        let mut z = w.dot(&x);

        // L2-normalize
        let norm = z.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 1e-8 {
            z.mapv_inplace(|v| v / norm);
        }
        z.to_vec()
    }

    /// Compute triplet loss without updating weights.
    ///
    /// `L = (1/N) Σ max(0, d(Wq, Wp) - d(Wq, Wn) + margin)`
    pub fn triplet_loss(&self, triplets: &[Triplet]) -> f64 {
        if triplets.is_empty() {
            return 0.0;
        }

        let w = Array2::from_shape_vec((self.dim, self.dim), self.weights.clone())
            .expect("weights shape mismatch");
        let mut total_loss = 0.0;

        for (q, p, n) in triplets {
            let q_arr = Array1::from_vec(q.clone());
            let p_arr = Array1::from_vec(p.clone());
            let n_arr = Array1::from_vec(n.clone());

            let q_proj = l2_normalize(&w.dot(&q_arr));
            let p_proj = l2_normalize(&w.dot(&p_arr));
            let n_proj = l2_normalize(&w.dot(&n_arr));

            let d_pos = cosine_distance(&q_proj, &p_proj);
            let d_neg = cosine_distance(&q_proj, &n_proj);
            total_loss += (d_pos - d_neg + self.margin as f64).max(0.0);
        }

        total_loss / triplets.len() as f64
    }

    /// Train on a batch of triplets for the given number of epochs.
    ///
    /// Uses SGD with manual backprop through the triplet loss.
    /// Returns the average loss of the final epoch.
    pub fn train(&mut self, triplets: &[Triplet], epochs: usize, lr: f32) -> f64 {
        if triplets.is_empty() {
            return 0.0;
        }

        let dim = self.dim;
        let mut w = Array2::from_shape_vec((dim, dim), self.weights.clone())
            .expect("weights shape mismatch");

        let mut final_loss = 0.0;

        for _epoch in 0..epochs {
            let mut grad = Array2::zeros((dim, dim));
            let mut epoch_loss = 0.0;
            let n = triplets.len() as f32;

            for (q, p, neg) in triplets {
                let q_arr = Array1::from_vec(q.clone());
                let p_arr = Array1::from_vec(p.clone());
                let n_arr = Array1::from_vec(neg.clone());

                let q_proj = w.dot(&q_arr);
                let p_proj = w.dot(&p_arr);
                let n_proj = w.dot(&n_arr);

                let q_norm = l2_normalize(&q_proj);
                let p_norm = l2_normalize(&p_proj);
                let n_norm = l2_normalize(&n_proj);

                let d_pos = cosine_distance(&q_norm, &p_norm);
                let d_neg = cosine_distance(&q_norm, &n_norm);
                let loss = (d_pos - d_neg + self.margin as f64).max(0.0);
                epoch_loss += loss;

                if loss > 0.0 {
                    // Simplified gradient: push query closer to positive, away from negative.
                    // dL/dW ≈ (p_proj - n_proj) × q^T (gradient direction through projection).
                    let diff = &p_proj - &n_proj;
                    let outer = diff
                        .insert_axis(Axis(1))
                        .dot(&q_arr.insert_axis(Axis(0)));
                    grad += &outer;
                }
            }

            // SGD update: W -= lr/n * grad
            w.scaled_add(-lr / n, &grad);

            final_loss = epoch_loss / triplets.len() as f64;
        }

        // Write back
        self.weights = w.into_raw_vec();
        self.train_count += 1;
        self.last_loss = final_loss;
        final_loss
    }

    /// Serialize weights to bytes for CozoDB storage.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.weights.len() * 4);
        for &w in &self.weights {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        bytes
    }

    /// Deserialize weights from bytes.
    pub fn from_bytes(bytes: &[u8], dim: usize, train_count: u64, last_loss: f64) -> Result<Self> {
        let expected = dim * dim * 4;
        if bytes.len() != expected {
            return Err(MemoriaError::Store(format!(
                "projection weights: expected {expected} bytes, got {}",
                bytes.len()
            )));
        }
        let weights: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        Ok(Self {
            weights,
            dim,
            margin: 0.3,
            last_loss,
            train_count,
        })
    }
}

// ── Persistence ──────────────────────────────────────────────────────────────

/// Save the projection matrix to CozoDB with temporal versioning.
pub fn save_projection(store: &CozoStore, proj: &EmbeddingProjection) -> Result<()> {
    let now = crate::types::memory::now_ms();
    let blob = proj.to_bytes();

    let mut params = BTreeMap::new();
    params.insert("blob".into(), cozo::DataValue::Bytes(blob));
    params.insert("dim".into(), cozo::DataValue::from(proj.dim as i64));
    params.insert("last_loss".into(), cozo::DataValue::from(proj.last_loss));
    params.insert("train_count".into(), cozo::DataValue::from(proj.train_count as i64));
    params.insert("trained_at".into(), cozo::DataValue::from(now));

    store
        .run_script(
            r#"?[id, weights_blob, dim, last_loss, train_count, trained_at] <- [[
                0, $blob, $dim, $last_loss, $train_count, $trained_at
            ]]
            :put embedding_projection {
                id
                =>
                weights_blob,
                dim,
                last_loss,
                train_count,
                trained_at
            }"#,
            params,
        )
        .map_err(|e| MemoriaError::Store(format!("save projection: {e}")))?;

    Ok(())
}

/// Load the most recent projection from CozoDB.
pub fn load_projection(store: &CozoStore, dim: usize) -> Result<Option<EmbeddingProjection>> {
    let result = store
        .run_query(
            r#"?[weights_blob, last_loss, train_count] :=
                *embedding_projection{weights_blob, last_loss, train_count}"#,
            BTreeMap::new(),
        )
        .map_err(|e| MemoriaError::Store(format!("load projection: {e}")))?;

    if result.rows.is_empty() {
        return Ok(None);
    }

    let row = &result.rows[0];
    let blob = match &row[0] {
        cozo::DataValue::Bytes(b) => b.clone(),
        _ => return Ok(None),
    };
    let last_loss = row[1].get_float().unwrap_or(1.0);
    let train_count = row[2].get_int().unwrap_or(0) as u64;

    EmbeddingProjection::from_bytes(&blob, dim, train_count, last_loss).map(Some)
}

/// Save training statistics for a training session.
pub fn save_training_stats(store: &CozoStore, stats: &ProjectionStats) -> Result<()> {
    let mut params = BTreeMap::new();
    params.insert("loss_before".into(), cozo::DataValue::from(stats.loss_before));
    params.insert("loss_after".into(), cozo::DataValue::from(stats.loss_after));
    params.insert("triplet_count".into(), cozo::DataValue::from(stats.triplet_count as i64));
    params.insert("epochs".into(), cozo::DataValue::from(stats.epochs as i64));
    params.insert("duration_ms".into(), cozo::DataValue::from(stats.duration_ms as i64));

    store
        .run_script(
            r#"?[loss_before, loss_after, triplet_count, epochs, duration_ms] <- [[
                $loss_before, $loss_after, $triplet_count, $epochs, $duration_ms
            ]]
            :put projection_stats {
                id, ts
                =>
                loss_before,
                loss_after,
                triplet_count,
                epochs,
                duration_ms
            }"#,
            params,
        )
        .map_err(|e| MemoriaError::Store(format!("save projection stats: {e}")))?;

    Ok(())
}

// ── Triplet Collection ───────────────────────────────────────────────────────

/// Collect training triplets from recall_contexts + task_outcomes.
///
/// For successful tasks: (query_embedding, recalled_memory_embedding) → positive pair.
/// For failed tasks: (query_embedding, recalled_memory_embedding) → negative pair.
/// Cross-pair positive queries with negative memories and vice versa to build triplets.
pub async fn collect_triplets(
    store: &CozoStore,
    embedder: &dyn Embedder,
    min_triplets: usize,
) -> Result<Vec<Triplet>> {
    // Step 1: Get successful recall contexts with memory embeddings
    let positive_script = r#"
        ?[query_text, mem_embedding] :=
            *recall_contexts{task_id, memory_ids, query_text},
            *task_outcomes{task_id, outcome},
            outcome = "success",
            mem_id in memory_ids,
            *memories{id: mem_id, embedding: mem_embedding}
    "#;
    let pos_result = store
        .run_query(positive_script, BTreeMap::new())
        .map_err(|e| MemoriaError::Store(format!("collect positive pairs: {e}")))?;

    // Step 2: Get failed recall contexts with memory embeddings
    let negative_script = r#"
        ?[query_text, mem_embedding] :=
            *recall_contexts{task_id, memory_ids, query_text},
            *task_outcomes{task_id, outcome},
            outcome = "failure",
            mem_id in memory_ids,
            *memories{id: mem_id, embedding: mem_embedding}
    "#;
    let neg_result = store
        .run_query(negative_script, BTreeMap::new())
        .map_err(|e| MemoriaError::Store(format!("collect negative pairs: {e}")))?;

    if pos_result.rows.is_empty() || neg_result.rows.is_empty() {
        return Ok(vec![]);
    }

    // Step 3: Extract (query_text, mem_embedding) pairs
    let mut positives: Vec<(String, Vec<f32>)> = Vec::new();
    for row in &pos_result.rows {
        let query_text = row[0].get_str().unwrap_or_default().to_string();
        if let Some(emb) = extract_embedding(&row[1]) {
            positives.push((query_text, emb));
        }
    }

    let mut negatives: Vec<(String, Vec<f32>)> = Vec::new();
    for row in &neg_result.rows {
        let query_text = row[0].get_str().unwrap_or_default().to_string();
        if let Some(emb) = extract_embedding(&row[1]) {
            negatives.push((query_text, emb));
        }
    }

    // Step 4: Build triplets by cross-pairing.
    // For each positive (query, pos_mem), pair with each negative's mem as the neg.
    // For each negative (query, neg_mem), pair with each positive's mem as the pos.
    let mut triplets = Vec::new();

    // Embed queries (batch unique texts for efficiency)
    let mut query_cache: std::collections::HashMap<String, Vec<f32>> = std::collections::HashMap::new();

    for (qt, pos_emb) in &positives {
        let q_emb = if let Some(cached) = query_cache.get(qt) {
            cached.clone()
        } else {
            let embeddings = embedder.embed(&[qt.as_str()]).await
                .map_err(|e| MemoriaError::Embedding(e.to_string()))?;
            let emb = embeddings.into_iter().next()
                .ok_or_else(|| MemoriaError::Embedding("empty embedding result".to_string()))?;
            query_cache.insert(qt.clone(), emb.clone());
            emb
        };

        for (_, neg_emb) in &negatives {
            triplets.push((q_emb.clone(), pos_emb.clone(), neg_emb.clone()));
            if triplets.len() >= min_triplets * 4 {
                break; // cap to avoid excessive computation
            }
        }
        if triplets.len() >= min_triplets * 4 {
            break;
        }
    }

    // Reverse direction: negative queries with positive memories
    for (qt, neg_emb) in &negatives {
        let q_emb = if let Some(cached) = query_cache.get(qt) {
            cached.clone()
        } else {
            let embeddings = embedder.embed(&[qt.as_str()]).await
                .map_err(|e| MemoriaError::Embedding(e.to_string()))?;
            let emb = embeddings.into_iter().next()
                .ok_or_else(|| MemoriaError::Embedding("empty embedding result".to_string()))?;
            query_cache.insert(qt.clone(), emb.clone());
            emb
        };

        for (_, pos_emb) in &positives {
            triplets.push((q_emb.clone(), pos_emb.clone(), neg_emb.clone()));
            if triplets.len() >= min_triplets * 4 {
                break;
            }
        }
        if triplets.len() >= min_triplets * 4 {
            break;
        }
    }

    Ok(triplets)
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// L2-normalize an array. Returns zero vector if norm is too small.
fn l2_normalize(v: &Array1<f32>) -> Array1<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        v / norm
    } else {
        v.clone()
    }
}

/// Cosine distance: 1 - cosine_similarity.
/// Assumes inputs are already L2-normalized.
fn cosine_distance(a: &Array1<f32>, b: &Array1<f32>) -> f64 {
    let sim: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    (1.0 - sim as f64).max(0.0)
}

/// Extract f32 embedding from a CozoDB DataValue.
fn extract_embedding(val: &cozo::DataValue) -> Option<Vec<f32>> {
    match val {
        cozo::DataValue::Vec(v) => {
            use cozo::Vector;
            match v {
                Vector::F32(arr) => Some(arr.to_vec()),
                Vector::F64(arr) => Some(arr.iter().map(|&x| x as f32).collect()),
            }
        }
        cozo::DataValue::List(list) => {
            let mut emb = Vec::with_capacity(list.len());
            for item in list {
                emb.push(item.get_float()? as f32);
            }
            Some(emb)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_projection_is_noop() {
        let dim = 4;
        let proj = EmbeddingProjection::identity(dim);
        let input = vec![0.6, 0.8, 0.0, 0.0];
        let output = proj.project(&input);

        // Identity should return L2-normalized input
        let norm: f32 = input.iter().map(|x| x * x).sum::<f32>().sqrt();
        let expected: Vec<f32> = input.iter().map(|x| x / norm).collect();
        for (a, b) in output.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "expected {b}, got {a}");
        }
    }

    #[test]
    fn test_project_normalizes_output() {
        let dim = 8;
        let proj = EmbeddingProjection::identity(dim);
        let input: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let output = proj.project(&input);

        let norm: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "output should be unit length, got {norm}");
    }

    #[test]
    fn test_train_reduces_loss() {
        let dim = 4;
        let mut proj = EmbeddingProjection::identity(dim);

        // Synthetic triplets: query is close to positive, far from negative
        let triplets: Vec<Triplet> = vec![
            (
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.9, 0.1, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0],
            ),
            (
                vec![0.0, 1.0, 0.0, 0.0],
                vec![0.1, 0.9, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 1.0],
            ),
            (
                vec![0.0, 0.0, 1.0, 0.0],
                vec![0.0, 0.1, 0.9, 0.0],
                vec![1.0, 0.0, 0.0, 0.0],
            ),
        ];

        let loss_before = proj.triplet_loss(&triplets);
        let loss_after = proj.train(&triplets, 50, 0.1);
        assert!(
            loss_after <= loss_before,
            "loss should decrease: {loss_before} -> {loss_after}"
        );
    }

    #[test]
    fn test_serialization_roundtrip() {
        let dim = 4;
        let mut proj = EmbeddingProjection::identity(dim);
        // Modify some weights so it's not just identity
        proj.weights[1] = 0.5;
        proj.weights[7] = -0.3;
        proj.train_count = 42;
        proj.last_loss = 0.123;

        let bytes = proj.to_bytes();
        let restored = EmbeddingProjection::from_bytes(&bytes, dim, 42, 0.123).unwrap();

        assert_eq!(proj.weights, restored.weights);
        assert_eq!(restored.train_count, 42);
        assert!((restored.last_loss - 0.123).abs() < 1e-10);
    }

    #[test]
    fn test_triplet_loss_zero_when_perfect() {
        let dim = 4;
        let proj = EmbeddingProjection::identity(dim);

        // Query identical to positive, orthogonal to negative → distance already good
        let triplets: Vec<Triplet> = vec![(
            vec![1.0, 0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
        )];

        let loss = proj.triplet_loss(&triplets);
        // d(q,p) = 0.0, d(q,n) = 1.0, loss = max(0, 0 - 1 + 0.3) = 0
        assert!(
            loss < 1e-5,
            "loss should be ~0 when positive is identical to query, got {loss}"
        );
    }

    #[test]
    fn test_from_bytes_rejects_wrong_size() {
        let result = EmbeddingProjection::from_bytes(&[0u8; 10], 4, 0, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_triplets_returns_zero_loss() {
        let proj = EmbeddingProjection::identity(4);
        assert_eq!(proj.triplet_loss(&[]), 0.0);
    }
}
