use uuid::Uuid;

use crate::error::Result;
use crate::store::CozoStore;
use crate::types::memory::now_ms;

/// Strengthen Hebbian associations between co-retrieved memories.
///
/// When memories are retrieved together in the same ask() call, their
/// pairwise association weights increase. This implements Hebb's rule:
/// "neurons that fire together wire together."
///
/// The weight update follows:
///   new_weight = old_weight + learning_rate × (1 - old_weight)
///
/// This saturates at 1.0 and produces diminishing returns for repeated
/// co-access (already strong associations grow slowly).
pub fn strengthen_associations(
    store: &CozoStore,
    memory_ids: &[Uuid],
    learning_rate: f64,
) -> Result<()> {
    if memory_ids.len() < 2 {
        return Ok(());
    }

    let now = now_ms();

    // Generate all unique pairs
    for i in 0..memory_ids.len() {
        for j in (i + 1)..memory_ids.len() {
            let a = memory_ids[i];
            let b = memory_ids[j];

            // Canonical ordering: smaller UUID first
            let (a, b) = if a < b { (a, b) } else { (b, a) };

            store.upsert_association(a, b, learning_rate, now)?;
            store.upsert_co_activation(a, b, now)?;
        }
    }

    Ok(())
}

/// Default Hebbian learning rate.
pub const DEFAULT_LEARNING_RATE: f64 = 0.1;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_pairs_for_single_memory() {
        let store = CozoStore::open_mem(4).unwrap();
        let id = Uuid::now_v7();
        // Should not error
        strengthen_associations(&store, &[id], DEFAULT_LEARNING_RATE).unwrap();
    }

    #[test]
    fn strengthens_pair_association() {
        let store = CozoStore::open_mem(4).unwrap();
        let a = Uuid::now_v7();
        let b = Uuid::now_v7();

        strengthen_associations(&store, &[a, b], DEFAULT_LEARNING_RATE).unwrap();

        // Check that association was created
        let (a_canon, b_canon) = if a < b { (a, b) } else { (b, a) };
        let weights = store.get_association_weights(&[a_canon], &[b_canon]).unwrap();
        assert!(!weights.is_empty(), "association should exist");
        assert!(weights[0].1 > 0.0, "weight should be positive");
    }

    #[test]
    fn repeated_strengthening_increases_weight() {
        let store = CozoStore::open_mem(4).unwrap();
        let a = Uuid::now_v7();
        let b = Uuid::now_v7();
        let (a_canon, b_canon) = if a < b { (a, b) } else { (b, a) };

        strengthen_associations(&store, &[a, b], DEFAULT_LEARNING_RATE).unwrap();
        let w1 = store.get_association_weights(&[a_canon], &[b_canon]).unwrap();

        strengthen_associations(&store, &[a, b], DEFAULT_LEARNING_RATE).unwrap();
        let w2 = store.get_association_weights(&[a_canon], &[b_canon]).unwrap();

        assert!(w2[0].1 > w1[0].1, "weight should increase with repeated co-access");
    }
}
