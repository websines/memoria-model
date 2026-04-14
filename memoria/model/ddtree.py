"""DDTree: Diffusion Draft Tree construction, compilation, and verification.

Builds a draft tree from the per-position marginal distributions of a single
block diffusion forward pass (DFlash), then verifies the entire tree in one
target-model forward pass using ancestor-only tree attention.

Algorithm 1 (Ringel & Romano, 2026): best-first heap search over rank tuples.
Given per-position top-K tokens and their log-probabilities, the heap
enumerates prefixes in descending order of factorized probability
q(ρ) = ∏ q_i^{(ρ_i)}. Each popped prefix generates at most 2 successors:
  - sibling: (ρ₁,...,ρ_{d-1}, ρ_d+1) — next-best token at current depth
  - child:   (ρ₁,...,ρ_d, 1)         — best token at next depth

Complexity: O(B log B) for B pops with heap of size O(B).

The resulting tree provably maximizes the expected acceptance length under
the drafter's factorized distribution Q(y|c,b) = ∏ q_i(y_i|c,b)
(Proposition 2 & 3, DDTree paper).

Reference: DDTree (Ringel & Romano — liranringel.github.io/ddtree/)
Reference: OPT-Tree (Wang et al. — TACL 2025) — adaptive draft tree for AR drafters
Reference: DFlash (Chen, Liang, Liu — arXiv:2602.06036) — block diffusion drafting
"""

import heapq
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor


@dataclass
class DDTreeResult:
    """Output of tree building: everything needed for compilation and walking.

    Attributes:
        node_token_ids: [N] token IDs for each tree node (excluding root).
        node_depths:    [N] depth of each node (1-indexed; root is depth 0).
        parents:        list of length (1 + N). parents[0] = -1 (root).
                        parents[i] = index of parent node for node i.
        child_maps:     list of dicts. child_maps[i] maps token_id → child_index.
                        Used by follow_verified_tree for the tree walk.
        visibility:     [1+N, 1+N] bool tensor. visibility[i,j] = True iff node j
                        is an ancestor of node i (or i == j). This IS the tree
                        attention mask for the verifier.
        node_count:     number of tree nodes (excluding root).
    """
    node_token_ids: Tensor          # [N] long, CPU
    node_depths: Tensor             # [N] long, CPU
    parents: list[int]              # length 1+N
    child_maps: list[dict[int, int]]  # length 1+N
    visibility: Tensor              # [1+N, 1+N] bool, CPU
    node_count: int


def build_ddtree(
    draft_logits: Tensor,
    budget: int,
) -> DDTreeResult:
    """Build an optimal draft tree from per-position draft logits.

    Implements Algorithm 1 from the DDTree paper: best-first enumeration
    of rank tuples using a max-heap over log-probabilities.

    The tree provably maximizes the surrogate expected acceptance length:
        max_{T:|T|≤B} E_{Y~Q}[α_T(Y)] = ∑_{u∈T} q(u|c,b)
    where Q is the drafter's factorized distribution (Proposition 2 & 3).

    Args:
        draft_logits: [L, vocab_size] raw logits from the draft head.
                      L = draft_length (number of future positions).
                      Each row is the marginal distribution at position i,
                      conditioned on context (c, b) but NOT on other positions.
        budget: maximum number of tree nodes (excluding root).
                Higher budget = more branches = longer acceptance but slower verify.
                Optimal is hardware-dependent; typically peaks at 256-512.

    Returns:
        DDTreeResult with tree structure for compilation and verification.
    """
    depth_limit = draft_logits.shape[0]

    if budget <= 0 or depth_limit == 0:
        return _empty_tree()

    # ── Top-K extraction ──────────────────────────────────────────────
    # Lemma 1 (DDTree paper): an optimal tree uses only the top-K tokens
    # at each depth, where K = min(B, |V|). Tokens ranked beyond B at
    # any position can never appear in the top-B prefixes by probability.
    top_k = min(budget, draft_logits.shape[-1])

    # Move to float32 for numerical stability in logsumexp
    logits_f32 = draft_logits.float()
    top_logits, top_token_ids = torch.topk(logits_f32, k=top_k, dim=-1)

    # Convert to log-probabilities: log_prob = logit - log(Z)
    # where Z = ∑ exp(logit) is the partition function per position
    log_z = torch.logsumexp(logits_f32, dim=-1, keepdim=True)
    top_log_probs = (top_logits - log_z).cpu().numpy().astype(np.float32)
    top_ids_np = top_token_ids.cpu().numpy().astype(np.int64)

    # ── Best-first heap search (Algorithm 1) ──────────────────────────
    # Heap elements: (-log_prob, rank_tuple, parent_idx, depth, last_rank, log_prob)
    # Negated log_prob because heapq is a min-heap; we want max-probability first.
    # rank_tuple is kept for tie-breaking (lexicographic order on ranks).
    first_logw = float(top_log_probs[0, 0])
    heap: list[tuple[float, tuple[int, ...], int, int, int, float]] = [
        (-first_logw, (0,), 0, 1, 0, first_logw)
    ]

    # Pre-allocate output arrays (budget is the max possible node count)
    node_token_ids_np = np.empty(budget, dtype=np.int64)
    node_depths_np = np.empty(budget, dtype=np.int64)
    parents_np = np.empty(budget + 1, dtype=np.int32)
    parents_np[0] = -1  # root has no parent
    child_maps: list[dict[int, int]] = [dict()]  # child_maps[0] = root's children
    node_count = 0

    while heap and node_count < budget:
        _, ranks, parent_index, depth, rank, logw = heapq.heappop(heap)

        # Resolve rank → token_id and record this node
        token_id = int(top_ids_np[depth - 1, rank])
        current_index = node_count + 1  # 0 is root, nodes are 1-indexed
        node_token_ids_np[node_count] = token_id
        node_depths_np[node_count] = depth
        parents_np[current_index] = parent_index
        child_maps.append(dict())
        child_maps[parent_index][token_id] = current_index
        node_count += 1

        # Push sibling: same depth, next rank (alternative token at this position)
        # Score delta: replace log q_d^{(ρ_d)} with log q_d^{(ρ_d+1)}
        if rank + 1 < top_k:
            sibling_logw = (
                logw
                - float(top_log_probs[depth - 1, rank])
                + float(top_log_probs[depth - 1, rank + 1])
            )
            sibling_ranks = ranks[:-1] + (rank + 1,)
            heapq.heappush(heap, (
                -sibling_logw, sibling_ranks, parent_index,
                depth, rank + 1, sibling_logw,
            ))

        # Push first child: extend to next depth with rank-1 token
        # Score delta: add log q_{d+1}^{(1)}
        if depth < depth_limit:
            child_logw = logw + float(top_log_probs[depth, 0])
            child_ranks = ranks + (0,)
            heapq.heappush(heap, (
                -child_logw, child_ranks, current_index,
                depth + 1, 0, child_logw,
            ))

    # ── Build ancestor-only visibility matrix ─────────────────────────
    # visibility[i, j] = True iff j is an ancestor of i (or i == j).
    # This is the tree attention mask: each node can attend to the root,
    # its ancestors, and itself — but NOT siblings or cousins.
    # Built by propagation: node i copies parent's row and sets self-bit.
    # Complexity: O(B²) in the worst case, but B is typically 256-512.
    total_length = 1 + node_count  # root + nodes
    visibility_np = np.zeros((total_length, total_length), dtype=np.bool_)
    visibility_np[0, 0] = True  # root sees itself
    for index in range(1, total_length):
        parent = int(parents_np[index])
        # Copy parent's visibility row (node inherits all ancestors)
        visibility_np[index, :index] = visibility_np[parent, :index]
        # Node sees itself
        visibility_np[index, index] = True

    return DDTreeResult(
        node_token_ids=torch.from_numpy(node_token_ids_np[:node_count]),
        node_depths=torch.from_numpy(node_depths_np[:node_count]),
        parents=parents_np[:total_length].tolist(),
        child_maps=child_maps,
        visibility=torch.from_numpy(visibility_np),
        node_count=node_count,
    )


def compile_ddtree(
    tree: DDTreeResult,
    root_token_id: int,
    start_position: int,
    past_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compile tree into verification tensors for a single target-model forward pass.

    Produces three tensors that the verifier consumes:
    1. input_ids:      [1, 1+N] token IDs (root + tree nodes)
    2. position_ids:   [1, 1+N] position indices (root=start, nodes=start+depth)
    3. attention_mask:  [1, 1, 1+N, past_length + 1+N] the full attention mask:
                       - left block [1+N, past_length]: all ones (attend to past context)
                       - right block [1+N, 1+N]: tree visibility (ancestor-only)

    The attention mask uses 0.0 for "attend" and dtype.min for "block", matching
    PyTorch's additive attention mask convention (added to QK^T before softmax).

    Args:
        tree: DDTreeResult from build_ddtree.
        root_token_id: the bonus token (guaranteed token from previous round).
        start_position: absolute position of the root in the sequence.
        past_length: number of past context tokens already in the KV cache.
        device: target device for output tensors.
        dtype: attention mask dtype (should match model dtype, e.g. bfloat16).

    Returns:
        (input_ids, position_ids, attention_mask) ready for target model forward.
    """
    current_length = 1 + tree.node_count  # root + nodes

    # ── Input IDs: root token followed by tree node tokens ────────────
    input_ids = torch.empty(1, current_length, dtype=torch.long, device=device)
    input_ids[0, 0] = root_token_id
    if tree.node_count > 0:
        input_ids[0, 1:] = tree.node_token_ids.to(device)

    # ── Position IDs: root at start_position, nodes at start + depth ──
    position_ids = torch.empty(1, current_length, dtype=torch.long, device=device)
    position_ids[0, 0] = start_position
    if tree.node_count > 0:
        position_ids[0, 1:] = tree.node_depths.to(device) + start_position

    # ── Attention mask: [1, 1, current_length, past_length + current_length] ──
    # Left block: all tree nodes attend to the full past context (causal is
    # guaranteed because past_length < start_position for all tree nodes).
    # Right block: tree visibility (ancestor-only attention within the tree).
    total_mask_width = past_length + current_length
    mask_blocked = torch.finfo(dtype).min

    attention_mask = torch.full(
        (1, 1, current_length, total_mask_width),
        mask_blocked, dtype=dtype, device=device,
    )

    # Left block: attend to all past context positions
    if past_length > 0:
        attention_mask[:, :, :, :past_length] = 0.0

    # Right block: tree visibility (ancestor-only)
    visibility_gpu = tree.visibility.to(device)
    tree_block = attention_mask[0, 0, :current_length, past_length:past_length + current_length]
    tree_block.masked_fill_(visibility_gpu, 0.0)

    return input_ids, position_ids, attention_mask


def follow_verified_tree(
    tree: DDTreeResult,
    posterior: Tensor,
    temperature: float = 0.0,
) -> tuple[list[int], int]:
    """Walk the verified tree: accept the longest matching path.

    Starting from the root, at each node the verifier's output distribution
    determines the next token. If that token matches a child in the tree,
    the walk continues down that branch. On the first mismatch (or leaf),
    the walk stops. The matched path is accepted; the first unmatched
    verifier token becomes the bonus token for the next round.

    This procedure is LOSSLESS: the target model uses its own decoding rule
    at every step, so DDTree preserves exactly the target model's output
    distribution (Proposition in DDTree paper, Section 4.4).

    Args:
        tree: DDTreeResult from build_ddtree (contains child_maps for walking).
        posterior: [1, 1+N, vocab] logits from the target model's verify pass.
                   One logit vector per tree position (root + nodes).
        temperature: sampling temperature (0.0 = greedy argmax).

    Returns:
        (accepted_indices, next_token_id):
        - accepted_indices: list of tree position indices on the accepted path
          (always starts with 0 = root). Length = acceptance length.
        - next_token_id: the first unmatched token from the verifier.
          This becomes the bonus token for the next decoding round.
    """
    # Sample from verifier logits at all tree positions
    if temperature < 1e-5:
        posterior_tokens = posterior[0].argmax(dim=-1).tolist()
    else:
        probs = torch.softmax(posterior[0] / temperature, dim=-1)
        posterior_tokens = torch.multinomial(probs, 1).squeeze(-1).tolist()

    accepted_indices = [0]  # root is always accepted
    current_index = 0
    next_token = int(posterior_tokens[current_index])

    # Walk: follow matching children until mismatch or leaf
    while next_token in tree.child_maps[current_index]:
        current_index = tree.child_maps[current_index][next_token]
        accepted_indices.append(current_index)
        next_token = int(posterior_tokens[current_index])

    return accepted_indices, next_token


def compute_tree_top_k_for_training(
    draft_logits: Tensor,
    target_tokens: Tensor,
    effective_k: int,
) -> Tensor:
    """Compute soft top-K recall signal for tree-aware training.

    For each position, measures whether the target token falls within the
    draft's top-K predictions. Positions where the target is outside top-K
    represent "tree misses" — the DDTree cannot possibly accept the correct
    token at that position regardless of budget.

    Returns a differentiable loss that pushes the target token's probability
    up when it's not in the top-K, providing direct gradient signal for
    improving tree coverage of the correct continuation.

    The loss at position i is:
        max(0, logsumexp(top_K_logits_i) - log_prob(target_i))
    When target is in top-K, this gap is ≤ log(K) (bounded, small gradient).
    When target is outside top-K, the gap is large → strong gradient signal.

    Args:
        draft_logits: [B, S, V] draft logits (BEFORE softmax).
        target_tokens: [B, S] target token IDs (-1 = padding/ignore).
        effective_k: K for top-K (typically min(budget, vocab_size)).

    Returns:
        scalar loss: mean recall gap across valid positions.
    """
    B, S, V = draft_logits.shape

    # Log-probabilities of the target token at each position
    log_probs = torch.log_softmax(draft_logits.float(), dim=-1)
    # Clamp target IDs for gather (handle -1 padding)
    safe_targets = target_tokens.clamp(min=0)
    target_log_prob = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)  # [B, S]

    # Log-sum-exp of top-K logits: the "ceiling" of the tree's coverage
    # If target is in top-K, target_log_prob is close to or within this
    # If target is outside top-K, target_log_prob << this
    clamped_k = min(effective_k, V)
    top_k_logits, _ = draft_logits.float().topk(clamped_k, dim=-1)  # [B, S, K]
    top_k_ceiling = torch.logsumexp(top_k_logits, dim=-1)  # [B, S]

    # Recall gap: how far the target is below the top-K ceiling
    # Clamped at 0 — no penalty when target is well within top-K
    recall_gap = (top_k_ceiling - target_log_prob).clamp(min=0.0)

    # Mask out padding positions
    valid = (target_tokens != -1).float()
    n_valid = valid.sum().clamp(min=1.0)

    return (recall_gap * valid).sum() / n_valid


def _empty_tree() -> DDTreeResult:
    """Return an empty tree (budget=0 or no draft positions)."""
    visibility = torch.zeros((1, 1), dtype=torch.bool)
    visibility[0, 0] = True
    return DDTreeResult(
        node_token_ids=torch.empty(0, dtype=torch.long),
        node_depths=torch.empty(0, dtype=torch.long),
        parents=[-1],
        child_maps=[dict()],
        visibility=visibility,
        node_count=0,
    )
