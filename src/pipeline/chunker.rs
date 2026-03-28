use text_splitter::TextSplitter;

/// Result of chunking a piece of text.
///
/// For short text (most agentic inputs), this is a single chunk.
/// For long text, it's semantically split into multiple chunks that
/// share a `group_id` and are linked via `chunk_sibling` edges.
#[derive(Debug, Clone)]
pub struct ChunkResult {
    /// The individual text chunks, split at semantic boundaries.
    pub chunks: Vec<String>,
    /// Whether the text was actually split (false = single chunk passthrough).
    pub was_split: bool,
}

/// Semantic text chunker for agentic memory.
///
/// Most agentic inputs are short (chat messages, tool outputs) and pass
/// through as a single chunk. Long text is split at semantic boundaries
/// (sentence/paragraph breaks) via the `text-splitter` crate.
///
/// Unlike hierarchical document chunkers, this produces a flat list —
/// the right model for agentic memory where inputs are already naturally
/// segmented by conversation turns.
pub struct SemanticChunker {
    /// Maximum characters per chunk. Text shorter than this passes through.
    max_chunk_size: usize,
}

impl SemanticChunker {
    pub fn new(max_chunk_size: usize) -> Self {
        Self { max_chunk_size }
    }

    /// Chunk text at semantic boundaries.
    ///
    /// - Short text (<= max_chunk_size): returns as-is (single chunk)
    /// - Long text: splits at sentence/paragraph boundaries
    pub fn chunk(&self, text: &str) -> ChunkResult {
        let text = text.trim();

        if text.is_empty() {
            return ChunkResult {
                chunks: Vec::new(),
                was_split: false,
            };
        }

        if text.len() <= self.max_chunk_size {
            return ChunkResult {
                chunks: vec![text.to_string()],
                was_split: false,
            };
        }

        let splitter = TextSplitter::new(self.max_chunk_size);
        let chunks: Vec<String> = splitter.chunks(text).map(|s| s.to_string()).collect();

        if chunks.is_empty() {
            return ChunkResult {
                chunks: vec![text.to_string()],
                was_split: false,
            };
        }

        let was_split = chunks.len() > 1;
        ChunkResult { chunks, was_split }
    }
}

impl Default for SemanticChunker {
    fn default() -> Self {
        Self {
            max_chunk_size: 1500,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_text_passes_through() {
        let chunker = SemanticChunker::default();
        let result = chunker.chunk("Alice works at Acme Corp");
        assert_eq!(result.chunks.len(), 1);
        assert!(!result.was_split);
        assert_eq!(result.chunks[0], "Alice works at Acme Corp");
    }

    #[test]
    fn empty_text_returns_empty() {
        let chunker = SemanticChunker::default();
        let result = chunker.chunk("");
        assert!(result.chunks.is_empty());
        assert!(!result.was_split);
    }

    #[test]
    fn whitespace_only_returns_empty() {
        let chunker = SemanticChunker::default();
        let result = chunker.chunk("   \n\t  ");
        assert!(result.chunks.is_empty());
    }

    #[test]
    fn long_text_splits_semantically() {
        let chunker = SemanticChunker::new(100);

        // Build text longer than 100 chars with clear sentence boundaries
        let text = "The quick brown fox jumps over the lazy dog. \
                     This is a second sentence that adds more content to the text. \
                     And here is a third sentence to make the text even longer than the limit. \
                     We need enough text to trigger splitting at semantic boundaries.";

        let result = chunker.chunk(text);
        assert!(result.was_split, "text should have been split");
        assert!(result.chunks.len() > 1, "expected multiple chunks");

        // No chunk should exceed max size
        for chunk in &result.chunks {
            assert!(
                chunk.len() <= 100,
                "chunk exceeds max size: {} chars",
                chunk.len()
            );
        }

        // All content should be preserved
        let joined: String = result.chunks.join("");
        let original_chars: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
        let joined_chars: Vec<char> = joined.chars().filter(|c| !c.is_whitespace()).collect();
        assert_eq!(original_chars.len(), joined_chars.len(), "no content should be lost");
    }

    #[test]
    fn text_at_boundary_is_not_split() {
        let chunker = SemanticChunker::new(50);
        let text = "Short enough to fit in one chunk easily."; // 40 chars
        let result = chunker.chunk(text);
        assert_eq!(result.chunks.len(), 1);
        assert!(!result.was_split);
    }
}
