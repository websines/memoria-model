//! Hierarchical chunking — sentence → paragraph → section → document.
//!
//! Each level is stored as a memory in CozoDB, linked via edges with
//! kind="chunk_parent". Retrieval matches at sentence level via HNSW,
//! then navigates up for broader context.

use uuid::Uuid;

use crate::error::Result;
use crate::store::CozoStore;
use crate::types::memory::Memory;

/// A hierarchical chunk with its level in the document tree.
#[derive(Debug, Clone)]
pub struct HierarchicalChunk {
    pub content: String,
    pub level: ChunkLevel,
    pub children: Vec<usize>, // indices into the chunks vec
}

/// Level in the chunk hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkLevel {
    Sentence,
    Paragraph,
    Section,
    Document,
}

impl ChunkLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Sentence => "chunk.sentence",
            Self::Paragraph => "chunk.paragraph",
            Self::Section => "chunk.section",
            Self::Document => "chunk.document",
        }
    }
}

/// Build a hierarchical chunk tree from a document.
///
/// Splits text into sentences, groups into paragraphs (by double newline),
/// groups into sections (by markdown headers or large gaps), and wraps
/// in a document node.
pub fn build_hierarchy(text: &str) -> Vec<HierarchicalChunk> {
    let mut chunks = Vec::new();

    if text.trim().is_empty() {
        return chunks;
    }

    // Split into paragraphs (double newline separated)
    let paragraphs: Vec<&str> = text
        .split("\n\n")
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .collect();

    if paragraphs.is_empty() {
        return chunks;
    }

    // Detect sections (paragraphs starting with # or significantly separated)
    let mut sections: Vec<Vec<&str>> = Vec::new();
    let mut current_section: Vec<&str> = Vec::new();

    for para in &paragraphs {
        if para.starts_with('#') && !current_section.is_empty() {
            sections.push(current_section.clone());
            current_section.clear();
        }
        current_section.push(para);
    }
    if !current_section.is_empty() {
        sections.push(current_section);
    }

    // Build bottom-up: sentences → paragraphs → sections → document
    let mut section_indices = Vec::new();

    for section_paras in &sections {
        let mut para_indices = Vec::new();

        for para in section_paras {
            // Split paragraph into sentences
            let sentences = split_sentences(para);
            let mut sentence_indices = Vec::new();

            for sentence in &sentences {
                let idx = chunks.len();
                chunks.push(HierarchicalChunk {
                    content: sentence.to_string(),
                    level: ChunkLevel::Sentence,
                    children: Vec::new(),
                });
                sentence_indices.push(idx);
            }

            // Create paragraph node
            let para_idx = chunks.len();
            chunks.push(HierarchicalChunk {
                content: para.to_string(),
                level: ChunkLevel::Paragraph,
                children: sentence_indices,
            });
            para_indices.push(para_idx);
        }

        // Create section node
        let section_content: Vec<&str> = section_paras.iter().copied().collect();
        let section_idx = chunks.len();
        chunks.push(HierarchicalChunk {
            content: section_content.join("\n\n"),
            level: ChunkLevel::Section,
            children: para_indices,
        });
        section_indices.push(section_idx);
    }

    // Create document node
    chunks.push(HierarchicalChunk {
        content: text.to_string(),
        level: ChunkLevel::Document,
        children: section_indices,
    });

    chunks
}

/// Simple sentence splitting. Splits on `.!?` followed by whitespace.
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        if (ch == '.' || ch == '!' || ch == '?') && current.len() > 10 {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current = String::new();
        }
    }

    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }

    sentences
}

/// Store hierarchical chunks in CozoDB with parent-child edges.
///
/// Each chunk becomes a memory, and parent-child relationships are
/// stored as edges with kind="chunk_parent".
///
/// Returns (chunk_index, memory_id) pairs for all stored chunks.
pub fn store_hierarchical_chunks(
    store: &CozoStore,
    chunks: &[HierarchicalChunk],
    embeddings: &[Vec<f32>],
    namespace: &str,
) -> Result<Vec<(usize, Uuid)>> {
    assert_eq!(chunks.len(), embeddings.len());

    let mut index_to_id: Vec<(usize, Uuid)> = Vec::with_capacity(chunks.len());

    // First pass: store all chunks as memories
    for (i, (chunk, embedding)) in chunks.iter().zip(embeddings.iter()).enumerate() {
        let mut memory = Memory::new(chunk.level.as_str(), &chunk.content, embedding.clone());
        memory.namespace = namespace.to_string();
        let id = memory.id;
        store.insert_memory(&memory)?;
        index_to_id.push((i, id));
    }

    // Second pass: create parent-child edges
    for (i, chunk) in chunks.iter().enumerate() {
        let parent_id = index_to_id[i].1;
        for &child_idx in &chunk.children {
            let child_id = index_to_id[child_idx].1;
            store.insert_edge(child_id, parent_id, "chunk_parent", 1.0)?;
        }
    }

    Ok(index_to_id)
}

/// Navigate up from a sentence-level memory to its parent context.
///
/// Returns memory IDs at each level: paragraph, section, document.
pub fn get_parent_chain(store: &CozoStore, memory_id: Uuid) -> Result<Vec<Uuid>> {
    let mut params = std::collections::BTreeMap::new();
    params.insert("id".into(), cozo::DataValue::from(memory_id.to_string()));

    let result = store.run_query(
        r#"
        parent[child, parent] := *edges{source: child, target: parent, kind},
                                  kind == "chunk_parent"
        chain[node] := node = to_uuid($id)
        chain[parent] := chain[child], parent[child, parent]
        ?[node] := chain[node], node != to_uuid($id)
        "#,
        params,
    )?;

    result
        .rows
        .iter()
        .map(|row| crate::store::cozo::parse_uuid_pub(&row[0]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_hierarchy_empty() {
        let chunks = build_hierarchy("");
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_build_hierarchy_single_sentence() {
        let chunks = build_hierarchy("Alice works at Acme Corp.");
        assert!(!chunks.is_empty());

        // Should have sentence + paragraph + section + document
        let levels: Vec<ChunkLevel> = chunks.iter().map(|c| c.level).collect();
        assert!(levels.contains(&ChunkLevel::Sentence));
        assert!(levels.contains(&ChunkLevel::Document));
    }

    #[test]
    fn test_build_hierarchy_multi_paragraph() {
        let text = "First paragraph with sentences. It has two sentences.\n\n\
                    Second paragraph here. Also two sentences.";
        let chunks = build_hierarchy(text);

        let para_count = chunks
            .iter()
            .filter(|c| c.level == ChunkLevel::Paragraph)
            .count();
        assert_eq!(para_count, 2, "should create 2 paragraphs");
    }

    #[test]
    fn test_build_hierarchy_with_sections() {
        let text = "# Introduction\n\nFirst paragraph.\n\n# Methods\n\nSecond paragraph.";
        let chunks = build_hierarchy(text);

        let section_count = chunks
            .iter()
            .filter(|c| c.level == ChunkLevel::Section)
            .count();
        assert_eq!(section_count, 2, "should create 2 sections from markdown headers");
    }

    #[test]
    fn test_store_and_navigate_hierarchy() {
        let store = CozoStore::open_mem(4).unwrap();

        let text = "Alice works at Acme Corp. She is an engineer.\n\nBob manages the team.";
        let chunks = build_hierarchy(text);

        // Create fake embeddings
        let embeddings: Vec<Vec<f32>> = chunks.iter().map(|_| vec![0.1, 0.2, 0.3, 0.4]).collect();

        let stored = store_hierarchical_chunks(&store, &chunks, &embeddings, "default").unwrap();
        assert_eq!(stored.len(), chunks.len());

        // Navigate up from a sentence
        let sentence_id = stored
            .iter()
            .find(|(i, _)| chunks[*i].level == ChunkLevel::Sentence)
            .map(|(_, id)| *id)
            .unwrap();

        let parents = get_parent_chain(&store, sentence_id).unwrap();
        assert!(
            !parents.is_empty(),
            "sentence should have parent nodes (paragraph, section, document)"
        );
    }

    #[test]
    fn test_split_sentences() {
        let text = "First sentence. Second sentence. Third one!";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 3);
    }

    #[test]
    fn test_chunk_level_as_str() {
        assert_eq!(ChunkLevel::Sentence.as_str(), "chunk.sentence");
        assert_eq!(ChunkLevel::Document.as_str(), "chunk.document");
    }
}
