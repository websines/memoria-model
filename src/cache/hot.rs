use dashmap::DashMap;
use std::time::Instant;

use crate::types::memory::{Memory, MemoryId};

/// Entry in the hot cache with access tracking for LRU eviction.
struct CacheEntry {
    memory: Memory,
    last_accessed: Instant,
    access_count: u32,
}

/// A concurrent hot cache for recently accessed memories.
///
/// Uses DashMap for lock-free concurrent reads. Entries are evicted
/// based on a combination of recency and access count (LRU-like).
pub struct HotCache {
    entries: DashMap<MemoryId, CacheEntry>,
    max_entries: usize,
}

impl HotCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: DashMap::with_capacity(max_entries),
            max_entries,
        }
    }

    /// Get a memory from the cache, updating access metadata.
    pub fn get(&self, id: &MemoryId) -> Option<Memory> {
        self.entries.get_mut(id).map(|mut entry| {
            entry.last_accessed = Instant::now();
            entry.access_count += 1;
            entry.memory.clone()
        })
    }

    /// Insert or update a memory in the cache.
    pub fn insert(&self, memory: Memory) {
        if self.entries.len() >= self.max_entries && !self.entries.contains_key(&memory.id) {
            self.evict_one();
        }

        self.entries.insert(
            memory.id,
            CacheEntry {
                memory,
                last_accessed: Instant::now(),
                access_count: 1,
            },
        );
    }

    /// Remove a memory from the cache.
    pub fn remove(&self, id: &MemoryId) -> Option<Memory> {
        self.entries.remove(id).map(|(_, entry)| entry.memory)
    }

    /// Invalidate the entire cache.
    pub fn clear(&self) {
        self.entries.clear();
    }

    /// Number of entries currently in the cache.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Evict the least recently used entry.
    fn evict_one(&self) {
        let oldest = self
            .entries
            .iter()
            .filter(|entry| !entry.memory.pinned) // Never evict pinned memories
            .min_by_key(|entry| entry.last_accessed)
            .map(|entry| *entry.key());

        if let Some(key) = oldest {
            self.entries.remove(&key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::memory::Memory;

    fn make_memory(content: &str) -> Memory {
        Memory::new("test", content, vec![0.1, 0.2, 0.3, 0.4])
    }

    #[test]
    fn test_insert_and_get() {
        let cache = HotCache::new(10);
        let mem = make_memory("hello");
        let id = mem.id;

        cache.insert(mem);
        let retrieved = cache.get(&id);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content, "hello");
    }

    #[test]
    fn test_cache_miss() {
        let cache = HotCache::new(10);
        let id = uuid::Uuid::now_v7();
        assert!(cache.get(&id).is_none());
    }

    #[test]
    fn test_remove() {
        let cache = HotCache::new(10);
        let mem = make_memory("hello");
        let id = mem.id;

        cache.insert(mem);
        assert!(cache.remove(&id).is_some());
        assert!(cache.get(&id).is_none());
    }

    #[test]
    fn test_eviction_at_capacity() {
        let cache = HotCache::new(2);

        let m1 = make_memory("first");
        let m2 = make_memory("second");
        let m3 = make_memory("third");

        cache.insert(m1);
        cache.insert(m2);
        assert_eq!(cache.len(), 2);

        cache.insert(m3);
        // Should still be at capacity (one evicted)
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_pinned_not_evicted() {
        let cache = HotCache::new(2);

        let mut m1 = make_memory("pinned");
        m1.pinned = true;
        let id1 = m1.id;

        let m2 = make_memory("second");
        let m3 = make_memory("third");

        cache.insert(m1);
        cache.insert(m2);
        cache.insert(m3);

        // Pinned memory should survive eviction
        assert!(cache.get(&id1).is_some());
    }

    #[test]
    fn test_clear() {
        let cache = HotCache::new(10);
        cache.insert(make_memory("a"));
        cache.insert(make_memory("b"));
        assert_eq!(cache.len(), 2);
        cache.clear();
        assert!(cache.is_empty());
    }
}
