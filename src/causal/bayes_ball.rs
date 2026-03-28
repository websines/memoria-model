//! Bayes-Ball algorithm for d-separation on petgraph DAGs.
//!
//! Implements the Bayes-Ball algorithm (Shachter 1998) which correctly handles:
//! - **Chains**: A→B→C — A and C are d-separated given B
//! - **Forks**: A←B→C — A and C are d-separated given B
//! - **Colliders**: A→B←C — A and C are d-separated *without* conditioning on B,
//!   but d-*connected* when conditioning on B or any descendant of B
//!
//! This replaces the previous Datalog-based d-separation which used simple
//! reachability and did not handle colliders correctly.

use std::collections::{HashSet, VecDeque};

use petgraph::graph::DiGraph;
use petgraph::graph::NodeIndex;
use petgraph::Direction;
use uuid::Uuid;

/// Check if X and Y are d-separated given conditioning set Z.
///
/// Returns `true` if X and Y are conditionally independent given Z
/// (i.e., no active path exists between them).
///
/// The algorithm traverses the graph tracking both the current node
/// and whether we arrived "from a child" (upward) or "from a parent"
/// (downward). This directionality is what makes Bayes-Ball handle
/// colliders correctly.
pub fn d_separated(
    graph: &DiGraph<Uuid, f64>,
    x: NodeIndex,
    y: NodeIndex,
    z: &HashSet<NodeIndex>,
) -> bool {
    // Pre-compute the set of nodes that are ancestors of Z.
    // Colliders are "activated" when they or any descendant are in Z.
    let z_or_descendant = ancestors_of_set(graph, z);

    let mut visited: HashSet<(NodeIndex, bool)> = HashSet::new();
    let mut queue: VecDeque<(NodeIndex, bool)> = VecDeque::new();

    // Start from X, going both directions
    queue.push_back((x, true)); // as if arrived from child (going up)
    queue.push_back((x, false)); // as if arrived from parent (going down)

    while let Some((node, from_child)) = queue.pop_front() {
        if !visited.insert((node, from_child)) {
            continue;
        }
        if node == y {
            return false; // reachable → not d-separated
        }

        let in_z = z.contains(&node);

        if from_child && !in_z {
            // Non-collider, not conditioned: signal passes through
            // Pass up to parents
            for parent in graph.neighbors_directed(node, Direction::Incoming) {
                queue.push_back((parent, true));
            }
            // Pass down to children
            for child in graph.neighbors_directed(node, Direction::Outgoing) {
                queue.push_back((child, false));
            }
        }

        if !from_child && !in_z {
            // Arrived from parent, not conditioned: pass down only
            for child in graph.neighbors_directed(node, Direction::Outgoing) {
                queue.push_back((child, false));
            }
        }

        if from_child && in_z {
            // Conditioned non-collider from child: blocked downward,
            // but pass up to parents (explaining away at ancestor level)
            for parent in graph.neighbors_directed(node, Direction::Incoming) {
                queue.push_back((parent, true));
            }
        }

        if !from_child && in_z {
            // Conditioned node from parent: blocked (non-collider conditioned)
            // No propagation
        }

        // Collider case: if node is a collider (has ≥2 parents) and
        // node or any descendant is in Z, it's activated
        if from_child {
            // Already handled above (non-collider cases)
        } else {
            // If we came from a parent and this is a collider...
            let parent_count = graph.neighbors_directed(node, Direction::Incoming).count();
            if parent_count >= 2 && z_or_descendant.contains(&node) {
                // Collider is activated: pass up to all parents
                for parent in graph.neighbors_directed(node, Direction::Incoming) {
                    queue.push_back((parent, true));
                }
            }
        }
    }

    true // Y not reachable → d-separated
}

/// Compute the set of all ancestors of Z (including Z itself).
///
/// A collider is activated if it or any of its descendants are in Z.
/// We compute this by traversing upward from each node in Z.
fn ancestors_of_set(
    graph: &DiGraph<Uuid, f64>,
    z: &HashSet<NodeIndex>,
) -> HashSet<NodeIndex> {
    let mut ancestors = z.clone();
    let mut queue: VecDeque<NodeIndex> = z.iter().copied().collect();

    // Traverse upward (to parents) from every node in Z
    while let Some(node) = queue.pop_front() {
        for parent in graph.neighbors_directed(node, Direction::Incoming) {
            if ancestors.insert(parent) {
                queue.push_back(parent);
            }
        }
    }

    ancestors
}

/// Convenience wrapper: check if two UUIDs are d-separated in a CausalGraph.
pub fn d_separated_by_uuid(
    graph: &super::graph::CausalGraph,
    x: Uuid,
    y: Uuid,
    z: &[Uuid],
) -> bool {
    let x_node = match graph.get_node(&x) {
        Some(n) => n,
        None => return true, // not in graph → trivially d-separated
    };
    let y_node = match graph.get_node(&y) {
        Some(n) => n,
        None => return true,
    };
    let z_set: HashSet<NodeIndex> = z
        .iter()
        .filter_map(|id| graph.get_node(id))
        .collect();

    d_separated(&graph.graph, x_node, y_node, &z_set)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_graph(nodes: usize, edges: &[(usize, usize)]) -> (DiGraph<Uuid, f64>, Vec<NodeIndex>) {
        let mut g = DiGraph::new();
        let ids: Vec<NodeIndex> = (0..nodes).map(|_| g.add_node(Uuid::now_v7())).collect();
        for &(from, to) in edges {
            g.add_edge(ids[from], ids[to], 1.0);
        }
        (g, ids)
    }

    #[test]
    fn test_chain_unconditional() {
        // A → B → C : A and C are NOT d-separated (path exists)
        let (g, n) = make_graph(3, &[(0, 1), (1, 2)]);
        assert!(!d_separated(&g, n[0], n[2], &HashSet::new()));
    }

    #[test]
    fn test_chain_conditioned() {
        // A → B → C : A and C ARE d-separated given B
        let (g, n) = make_graph(3, &[(0, 1), (1, 2)]);
        let z: HashSet<_> = [n[1]].into();
        assert!(d_separated(&g, n[0], n[2], &z));
    }

    #[test]
    fn test_fork_unconditional() {
        // A ← B → C : A and C are NOT d-separated (path through B)
        let (g, n) = make_graph(3, &[(1, 0), (1, 2)]);
        assert!(!d_separated(&g, n[0], n[2], &HashSet::new()));
    }

    #[test]
    fn test_fork_conditioned() {
        // A ← B → C : A and C ARE d-separated given B
        let (g, n) = make_graph(3, &[(1, 0), (1, 2)]);
        let z: HashSet<_> = [n[1]].into();
        assert!(d_separated(&g, n[0], n[2], &z));
    }

    #[test]
    fn test_collider_unconditional() {
        // A → B ← C : A and C ARE d-separated (collider blocks)
        let (g, n) = make_graph(3, &[(0, 1), (2, 1)]);
        assert!(d_separated(&g, n[0], n[2], &HashSet::new()));
    }

    #[test]
    fn test_collider_conditioned() {
        // A → B ← C : A and C are NOT d-separated given B (explaining away)
        let (g, n) = make_graph(3, &[(0, 1), (2, 1)]);
        let z: HashSet<_> = [n[1]].into();
        assert!(!d_separated(&g, n[0], n[2], &z));
    }

    #[test]
    fn test_collider_descendant_conditioned() {
        // A → B ← C, B → D : conditioning on D also activates collider B
        let (g, n) = make_graph(4, &[(0, 1), (2, 1), (1, 3)]);
        let z: HashSet<_> = [n[3]].into();
        assert!(!d_separated(&g, n[0], n[2], &z));
    }

    #[test]
    fn test_disconnected() {
        // A → B, C → D : A and C are d-separated (no path)
        let (g, n) = make_graph(4, &[(0, 1), (2, 3)]);
        assert!(d_separated(&g, n[0], n[2], &HashSet::new()));
    }

    #[test]
    fn test_diamond() {
        // A → B → D, A → C → D : A and D are NOT d-separated unconditionally
        let (g, n) = make_graph(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
        assert!(!d_separated(&g, n[0], n[3], &HashSet::new()));
    }

    #[test]
    fn test_diamond_conditioned_on_both_mediators() {
        // A → B → D, A → C → D : A and D ARE d-separated given {B, C}
        let (g, n) = make_graph(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
        let z: HashSet<_> = [n[1], n[2]].into();
        assert!(d_separated(&g, n[0], n[3], &z));
    }

    #[test]
    fn test_same_node() {
        // X == Y: should return false (not d-separated from itself)
        let (g, n) = make_graph(1, &[]);
        assert!(!d_separated(&g, n[0], n[0], &HashSet::new()));
    }
}
