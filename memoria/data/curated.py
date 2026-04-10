"""Curated multi-tier dataset streaming for cognitive state training.

Replaces the generic 90% web text mix with datasets that genuinely require
persistent state (belief tracking, revision, causal chaining) for prediction.

Target: ~45% of training data requires persistent cross-context memory.

All datasets stream from HuggingFace — zero local disk footprint.
Failed sources gracefully redistribute weight to remaining sources.

Tier structure:
  Tier 1 (25%) — State essential: can't predict answers without cross-context memory
  Tier 2 (20%) — State helps: multi-hop, causal, fact verification
  Tier 3 (15%) — General language: FineWeb-Edu + code (fluency baseline)
  Tier 4 (15%) — Reasoning: NVIDIA math/science/code with CoT
  Tier 5 (10%) — Tool calling: multi-step agentic trajectories
  Tier 6  (5%) — Enhanced synthetic: belief tracking, causal, contradiction tasks
  Reserve (10%) — redistributed to Tiers 1-2

Reference: docs/DATASET_CURATION.md
"""

import random
import torch
from dataclasses import dataclass, field
from torch import Tensor
from typing import Callable, Iterator
from transformers import PreTrainedTokenizer
from datasets import load_dataset

from . import formatters as fmt
from .streaming import stream_fineweb_edu, stream_code
from .synthetic import generate_all_synthetic


@dataclass
class DataSource:
    """A single streaming dataset source."""
    name: str
    hf_id: str
    weight: float                        # relative weight in mix (normalized at runtime)
    format_fn: Callable[[dict], str]     # raw sample → continuous text
    tier: str = "general"
    config: str | None = None            # HF dataset config/subset name
    split: str = "train"
    sub_sources: list[dict] | None = None  # for datasets with multiple configs/splits


# ── Dataset Registry ──
# Weights are relative within each tier, then scaled by tier weight.
# Only includes datasets verified to stream successfully.

TIER_WEIGHTS = {
    "state_essential": 0.15,   # belief tracking, ToM, fact revision
    "state_helps": 0.10,       # multi-hop, causal, verification
    "code": 0.30,              # raw code + code reasoning + competitive programming
    "code_agent": 0.15,        # terminal agents, SWE trajectories, tool-use coding
    "reasoning": 0.10,         # math, cross-domain reasoning
    "tool_calling": 0.08,      # function calling, agentic
    "general": 0.05,           # web text baseline
    "synthetic": 0.02,         # belief/causal synthetic tasks
    # Note: weights are reference only — actual mix is determined by per-source weights
}

CURATED_SOURCES: list[DataSource] = [

    # ══════════════════════════════════════════════════════════
    # CODE & CODE REASONING (~30%)
    # Raw code, competitive programming, code instruction
    # ══════════════════════════════════════════════════════════

    # StarCoderData: massive raw code corpus (Python)
    DataSource(
        name="starcoderdata_python",
        hf_id="bigcode/starcoderdata",
        weight=0.06,
        format_fn=fmt.format_starcoderdata,
        tier="code",
        config="python",  # uses data_dir
    ),
    # StarCoderData: JavaScript
    DataSource(
        name="starcoderdata_js",
        hf_id="bigcode/starcoderdata",
        weight=0.03,
        format_fn=fmt.format_starcoderdata,
        tier="code",
        config="javascript",
    ),
    # StarCoderData: Rust + Go + TypeScript
    DataSource(
        name="starcoderdata_rust",
        hf_id="bigcode/starcoderdata",
        weight=0.02,
        format_fn=fmt.format_starcoderdata,
        tier="code",
        config="rust",
    ),
    # OpenCodeReasoning-2: competitive programming with reasoning (Python)
    DataSource(
        name="opencode_reasoning_py",
        hf_id="nvidia/OpenCodeReasoning-2",
        weight=0.04,
        format_fn=fmt.format_opencode_reasoning,
        tier="code",
        split="python",
    ),
    # OpenCodeReasoning-2: C++
    DataSource(
        name="opencode_reasoning_cpp",
        hf_id="nvidia/OpenCodeReasoning-2",
        weight=0.02,
        format_fn=fmt.format_opencode_reasoning,
        tier="code",
        split="cpp",
    ),
    # Nemotron SFT OpenCode: NVIDIA's curated code instruction data
    # All splits: general code, bash tool use, agent skills, question+tool
    DataSource(
        name="nemotron_opencode_general",
        hf_id="nvidia/Nemotron-SFT-OpenCode-v1",
        weight=0.015,
        format_fn=fmt.format_nemotron_opencode,
        tier="code",
        split="general",
    ),
    DataSource(
        name="nemotron_opencode_agent",
        hf_id="nvidia/Nemotron-SFT-OpenCode-v1",
        weight=0.01,
        format_fn=fmt.format_nemotron_opencode,
        tier="code",
        split="agent_skills",
    ),
    DataSource(
        name="nemotron_opencode_bash",
        hf_id="nvidia/Nemotron-SFT-OpenCode-v1",
        weight=0.01,
        format_fn=fmt.format_nemotron_opencode,
        tier="code",
        split="bash_only_tool",
    ),
    DataSource(
        name="nemotron_opencode_question_tool",
        hf_id="nvidia/Nemotron-SFT-OpenCode-v1",
        weight=0.005,
        format_fn=fmt.format_nemotron_opencode,
        tier="code",
        split="question_tool",
    ),
    # DeepCoder: competitive programming
    DataSource(
        name="deepcoder",
        hf_id="agentica-org/DeepCoder-Preview-Dataset",
        weight=0.03,
        format_fn=fmt.format_deepcoder,
        tier="code",
        config="codeforces",
        split="test",  # only split available
    ),
    # Open-R1 Codeforces: competitive programming
    DataSource(
        name="codeforces",
        hf_id="open-r1/codeforces",
        weight=0.02,
        format_fn=fmt.format_codeforces,
        tier="code",
    ),
    # High-Coder Reasoning Multi-Turn
    DataSource(
        name="high_coder",
        hf_id="Crownelius/High-Coder-Reasoning-Multi-Turn",
        weight=0.02,
        format_fn=fmt.format_high_coder,
        tier="code",
    ),
    # Tiny-Codes: small code snippets
    DataSource(
        name="tiny_codes",
        hf_id="nampdn-ai/tiny-codes",
        weight=0.02,
        format_fn=fmt.format_tiny_codes,
        tier="code",
    ),

    # ══════════════════════════════════════════════════════════
    # CODE AGENTS & TERMINAL (~15%)
    # Terminal coding, SWE trajectories, agentic code
    # ══════════════════════════════════════════════════════════

    # Nemotron Terminal Corpus (original NVIDIA): 366K terminal execution trajectories.
    # Data querying, model training, data processing, debugging, software engineering.
    # Dataset adapters (226K) + skill-based synthetic tasks (140K). CC-BY-4.0.
    # Nemotron Terminal Corpus: terminal execution trajectories.
    # dataset_adapters config has nested array issue with HF streaming — use skill splits.
    DataSource(
        name="nemotron_terminal_medium",
        hf_id="nvidia/Nemotron-Terminal-Corpus",
        weight=0.03,
        format_fn=fmt.format_nemotron_terminal,
        tier="code_agent",
        config="skill_based_medium",
    ),
    DataSource(
        name="nemotron_terminal_easy",
        hf_id="nvidia/Nemotron-Terminal-Corpus",
        weight=0.02,
        format_fn=fmt.format_nemotron_terminal,
        tier="code_agent",
        config="skill_based_easy",
    ),
    DataSource(
        name="nemotron_terminal_mixed",
        hf_id="nvidia/Nemotron-Terminal-Corpus",
        weight=0.01,
        format_fn=fmt.format_nemotron_terminal,
        tier="code_agent",
        config="skill_based_mixed",
    ),
    # SWE-agent trajectories: full software engineering agent runs
    DataSource(
        name="swe_trajectories",
        hf_id="nebius/SWE-agent-trajectories",
        weight=0.03,
        format_fn=fmt.format_swe_trajectories,
        tier="code_agent",
    ),
    # Nemotron RL Agentic SWE: software engineering pivots
    DataSource(
        name="nemotron_rl_swe",
        hf_id="nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1",
        weight=0.02,
        format_fn=fmt.format_nemotron_swe,
        tier="code_agent",
    ),
    # Nemotron-Agentic-v1: tool calling split
    # Nemotron-Agentic-v1 tool_calling: nested struct schema breaks default HF streaming.
    # Fix: load with trust_remote_code to bypass strict schema casting.
    DataSource(
        name="nemotron_agentic_tool",
        hf_id="nvidia/Nemotron-Agentic-v1",
        weight=0.02,
        format_fn=fmt.format_nemotron_agentic,
        tier="code_agent",
        split="tool_calling",
    ),
    # Nemotron-Agentic-v1: interactive agent split
    DataSource(
        name="nemotron_agentic_interactive",
        hf_id="nvidia/Nemotron-Agentic-v1",
        weight=0.02,
        format_fn=fmt.format_nemotron_agentic,
        tier="code_agent",
        split="interactive_agent",
    ),
    # Nemotron RL FC: function calling decision pivots
    DataSource(
        name="nemotron_rl_fc",
        hf_id="nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1",
        weight=0.01,
        format_fn=fmt.format_nemotron_rl_agentic,
        tier="code_agent",
    ),
    # Nemotron RL Conv Tool: conversational tool use pivots
    DataSource(
        name="nemotron_rl_conv_tool",
        hf_id="nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1",
        weight=0.01,
        format_fn=fmt.format_nemotron_rl_agentic,
        tier="code_agent",
    ),
    # Hermes Agent Reasoning Traces (quality-filtered): real multi-turn tool-calling
    # trajectories. 3.7K samples filtered for genuine deliberation: 63% self-correction
    # (vs 6% unfiltered), 95.9% verification, 43.7% alternative exploration.
    # Avg 32 turns, 18.5 tool calls, 581-word <think> depth (+40% vs original).
    # Key for: goal tracking, causal edges, refinement loop internalization.
    DataSource(
        name="hermes_agent_traces",
        hf_id="DJLougen/hermes-agent-traces-filtered",
        weight=0.04,
        format_fn=fmt.format_hermes_agent_traces,
        tier="code_agent",
    ),

    # ══════════════════════════════════════════════════════════
    # STATE ESSENTIAL (~15%)
    # Can't predict answers without cross-context memory
    # ══════════════════════════════════════════════════════════

    DataSource(
        name="babilong",
        hf_id="RMT-team/babilong-train-5k-samples",
        weight=0.06,
        format_fn=fmt.format_babilong,
        tier="state_essential",
        config="4k",
        split="qa1",
        sub_sources=[
            {"config": c, "split": s}
            for c in ("0k", "4k", "8k", "16k")
            for s in ("qa1", "qa2", "qa3", "qa5")
        ],
    ),
    DataSource(
        name="wikifactdiff",
        hf_id="Orange/WikiFactDiff",
        weight=0.05,
        format_fn=fmt.format_wikifactdiff,
        tier="state_essential",
        config="20210104-20230227_legacy",
    ),
    DataSource(
        name="exploretom",
        hf_id="facebook/ExploreToM",
        weight=0.04,
        format_fn=fmt.format_exploretom,
        tier="state_essential",
    ),

    # ══════════════════════════════════════════════════════════
    # STATE HELPS (~10%)
    # Multi-hop, causal, fact verification
    # ══════════════════════════════════════════════════════════

    DataSource(name="proofwriter", hf_id="tasksource/proofwriter",
               weight=0.02, format_fn=fmt.format_proofwriter, tier="state_helps"),
    DataSource(name="vitaminc", hf_id="tals/vitaminc",
               weight=0.015, format_fn=fmt.format_vitaminc, tier="state_helps"),
    DataSource(name="corr2cause", hf_id="causal-nlp/corr2cause",
               weight=0.015, format_fn=fmt.format_corr2cause, tier="state_helps"),
    DataSource(name="hotpotqa", hf_id="hotpotqa/hotpot_qa",
               weight=0.02, format_fn=fmt.format_hotpotqa, tier="state_helps", config="distractor"),
    DataSource(name="tgqa", hf_id="sxiong/TGQA",
               weight=0.01, format_fn=fmt.format_tgqa, tier="state_helps", config="TGQA_Story_TG_Trans"),
    DataSource(name="goalstep", hf_id="tasksource/goal-step-wikihow",
               weight=0.01, format_fn=fmt.format_goalstep, tier="state_helps", config="goal"),
    DataSource(name="ecare", hf_id="12ml/e-CARE",
               weight=0.01, format_fn=fmt.format_ecare, tier="state_helps"),
    DataSource(name="opentom", hf_id="SeacowX/OpenToM",
               weight=0.005, format_fn=fmt.format_opentom, tier="state_helps", split="Long"),

    # ══════════════════════════════════════════════════════════
    # REASONING (~10%)
    # Math, cross-domain reasoning
    # ══════════════════════════════════════════════════════════

    DataSource(name="openmath", hf_id="nvidia/OpenMathReasoning",
               weight=0.03, format_fn=fmt.format_openmath, tier="reasoning", split="cot"),
    DataSource(name="crossthink", hf_id="nvidia/Nemotron-CrossThink",
               weight=0.02, format_fn=fmt.format_crossthink, tier="reasoning", split="train_qa"),
    DataSource(name="nemotron_mind", hf_id="nvidia/Nemotron-MIND",
               weight=0.02, format_fn=fmt.format_nemotron_mind, tier="reasoning"),
    DataSource(name="gsm8k", hf_id="openai/gsm8k",
               weight=0.01, format_fn=fmt.format_gsm8k, tier="reasoning", config="main"),
    DataSource(name="openmath_instruct", hf_id="nvidia/OpenMathInstruct-1",
               weight=0.02, format_fn=fmt.format_openmath_instruct, tier="reasoning"),
    # Nemotron-Science-v1: 226K scientific reasoning QA (GPQA-style MCQA + chemistry)
    # Genetics, pharmacogenomics, CRISPR, RNA-seq, synthetic biology.
    # Reasoning traces teach structured scientific thinking.
    DataSource(name="nemotron_science", hf_id="nvidia/Nemotron-Science-v1",
               weight=0.03, format_fn=fmt.format_nemotron_science, tier="reasoning",
               split="MCQ"),  # available: MCQ, RQA
    # OpenR1-Math-220k: 220K math problems with multiple DeepSeek R1 reasoning traces.
    # 2-4 verified solutions per problem. Dual verification (Math Verify + Llama-3.3-70B).
    # 16K token budget per generation. Olympiad + K-12 + competition sources.
    DataSource(name="openr1_math", hf_id="open-r1/OpenR1-Math-220k",
               weight=0.03, format_fn=fmt.format_openr1_math, tier="reasoning",
               config="default"),

    # ══════════════════════════════════════════════════════════
    # TOOL CALLING (~8%)
    # Function calling, multi-turn tools
    # ══════════════════════════════════════════════════════════

    DataSource(name="xlam_fc_60k", hf_id="Salesforce/xlam-function-calling-60k",
               weight=0.02, format_fn=fmt.format_xlam_fc, tier="tool_calling"),
    DataSource(name="hermes_fc", hf_id="NousResearch/hermes-function-calling-v1",
               weight=0.015, format_fn=fmt.format_hermes_fc, tier="tool_calling"),
    # toolbench: tuandunghcmut/toolbench-v1 removed from Hub — dropped
    DataSource(name="when2call", hf_id="nvidia/When2Call",
               weight=0.01, format_fn=fmt.format_when2call, tier="tool_calling",
               split="mcq"),  # available: llm_judge, mcq
    DataSource(name="xlam_irrelevance", hf_id="MadeAgents/xlam-irrelevance-7.5k",
               weight=0.005, format_fn=fmt.format_xlam_irrelevance, tier="tool_calling"),
    DataSource(name="toolace", hf_id="Team-ACE/ToolACE",
               weight=0.005, format_fn=fmt.format_toolace, tier="tool_calling"),
    DataSource(name="tool_multiturn", hf_id="interstellarninja/tool-use-multiturn-reasoning",
               weight=0.01, format_fn=fmt.format_tool_multiturn, tier="tool_calling"),
    DataSource(name="toolmind", hf_id="Nanbeige/ToolMind",
               weight=0.005, format_fn=fmt.format_toolmind, tier="tool_calling", split="open_datasets"),

    # ══════════════════════════════════════════════════════════
    # GENERAL (~5%)
    # Web text baseline for language fluency
    # ══════════════════════════════════════════════════════════

    DataSource(name="tiny_webtext", hf_id="nampdn-ai/tiny-webtext",
               weight=0.01, format_fn=fmt.format_tiny_webtext, tier="general"),

    # ══════════════════════════════════════════════════════════
    # SPECIALIZED PRETRAINING (~5%)
    # High-quality synthetic pretraining for code concepts, logic, general knowledge
    # ══════════════════════════════════════════════════════════

    # Nemotron-Pretraining-Specialized-v1.1: 9.3B tokens across 5 domains.
    # Code Concepts (15M, concept taxonomy), Formal Logic (491K), Economics (345K),
    # MMLU-style MCQ (3.5M). Generated by gpt-oss/Qwen3/DeepSeek-v3. CC-BY-4.0.
    DataSource(
        name="nemotron_pt_code_concepts",
        hf_id="nvidia/Nemotron-Pretraining-Specialized-v1.1",
        weight=0.02,
        format_fn=fmt.format_nemotron_pretraining,
        tier="code",
        config="Nemotron-Pretraining-Code-Concepts",
    ),
    DataSource(
        name="nemotron_pt_formal_logic",
        hf_id="nvidia/Nemotron-Pretraining-Specialized-v1.1",
        weight=0.01,
        format_fn=fmt.format_nemotron_pretraining,
        tier="reasoning",
        config="Nemotron-Pretraining-Formal-Logic",
    ),
    # nemotron_pt_mcq: Nemotron-Pretraining-Multiple-Choice parquet files cause
    # fsspec "Bad file descriptor" in background threads — kills DDP training.
    # Dropped. 1% weight redistributed to other general sources.

    # ══════════════════════════════════════════════════════════
    # SFT-QUALITY REASONING (Phase 3a data, low weight in PT mix)
    # Elite reasoning traces for refinement loop internalization
    # ══════════════════════════════════════════════════════════

    # GPT-5.4 XHIGH Reasoning: 2.7K elite samples, ~12,600 char thinking traces.
    # 60+ expert domains (math, code, science, topology, CRISPR, ZK-proofs...).
    # Small but extremely dense — pure SFT quality in pretraining mix.
    DataSource(
        name="xhigh_reasoning",
        hf_id="vanty120/Gpt-5.4-Xhigh-Reasoning-2000x",
        weight=0.01,
        format_fn=fmt.format_xhigh_reasoning,
        tier="reasoning",
    ),
]


# ── Stream Construction ──

def _load_parquet_fallback(source: DataSource) -> Iterator[dict]:
    """Load dataset by directly streaming parquet files — bypasses schema inference.

    Used when load_dataset() fails with "Couldn't cast array of type struct"
    errors on datasets with inconsistent nested schemas (e.g. Nemotron-Agentic
    tool_calling split). Loading raw parquet avoids the strict type casting.
    """
    from huggingface_hub import HfApi
    api = HfApi()
    files = api.list_repo_files(source.hf_id, repo_type="dataset")
    split = source.split or "train"
    config_prefix = f"{source.config}/" if source.config else ""

    # Find parquet files matching config + split
    parquet_files = [
        f"hf://datasets/{source.hf_id}/{f}"
        for f in files
        if f.endswith('.parquet') and split in f and (not config_prefix or config_prefix in f)
    ]
    if not parquet_files:
        # Broader search — any parquet with split name
        parquet_files = [
            f"hf://datasets/{source.hf_id}/{f}"
            for f in files
            if f.endswith('.parquet') and split in f
        ]
    if not parquet_files:
        raise RuntimeError(f"No parquet files found for {source.hf_id} split={split}")

    ds = load_dataset("parquet", data_files=parquet_files, streaming=True, split="train")
    return iter(ds)


def _load_hf_stream(source: DataSource) -> Iterator[dict]:
    """Load a single HuggingFace dataset as a streaming iterator.

    For sources with sub_sources (e.g. BABILong with multiple tasks),
    round-robins across all sub-source streams.

    Sources marked with _use_parquet_fallback (set during probe phase when
    schema cast errors are detected) use direct parquet loading.
    """
    if source.sub_sources:
        return _load_multi_stream(source)

    # If probe phase detected a schema issue, go straight to parquet
    if getattr(source, '_use_parquet_fallback', False):
        return _load_parquet_fallback(source)

    kwargs = dict(split=source.split, streaming=True)
    if source.config:
        if 'starcoderdata' in source.hf_id:
            kwargs['data_dir'] = source.config
        else:
            kwargs['name'] = source.config
    return iter(load_dataset(source.hf_id, **kwargs))


def _load_multi_stream(source: DataSource) -> Iterator[dict]:
    """Round-robin across multiple sub-source configs/splits."""
    streams: list[Iterator] = []
    for sub in source.sub_sources:
        try:
            kwargs = dict(streaming=True)
            kwargs['split'] = sub.get('split', source.split)
            config = sub.get('config', source.config)
            if config:
                kwargs['name'] = config
            streams.append(iter(load_dataset(source.hf_id, **kwargs)))
        except Exception:
            continue

    if not streams:
        raise RuntimeError(f"No sub-sources loaded for {source.name}")

    while streams:
        exhausted = []
        for i, stream in enumerate(streams):
            try:
                yield next(stream)
            except StopIteration:
                exhausted.append(i)
        # Remove exhausted streams in reverse order
        for i in reversed(exhausted):
            streams.pop(i)


def _text_stream_from_source(source: DataSource) -> Iterator[str]:
    """Yield formatted text strings from a single data source. Auto-restarts."""
    while True:
        try:
            hf_stream = _load_hf_stream(source)
        except Exception as e:
            print(f"WARNING: {source.name} stream restart failed: {e}")
            return

        yielded_any = False
        for sample in hf_stream:
            try:
                text = source.format_fn(sample)
                if text:
                    yielded_any = True
                    yield text
            except Exception:
                continue

        if not yielded_any:
            return  # dataset is empty or all samples malformed


class _ActiveSource:
    """Wraps a text stream with its weight and restart capability."""
    __slots__ = ('source', 'stream', 'weight', 'alive')

    def __init__(self, source: DataSource, weight: float):
        self.source = source
        self.stream = _text_stream_from_source(source)
        self.weight = weight
        self.alive = True

    def next_text(self) -> str | None:
        try:
            return next(self.stream)
        except StopIteration:
            self.alive = False
            return None


def curated_stream(
    tokenizer: PreTrainedTokenizer | None,
    seq_len: int = 2048,
    synthetic_data: list[str] | None = None,
    code_languages: list[str] | None = None,
    skip_documents: int = 0,
    byte_mode: bool = False,
) -> Iterator[dict[str, Tensor]]:
    """Stream sequences from the curated multi-tier dataset mix.

    Drop-in replacement for interleaved_stream() with state-essential data.

    Args:
        tokenizer: tokenizer for encoding (ignored in byte_mode)
        seq_len: sequence length (tokens in token mode, bytes in byte mode)
        synthetic_data: pre-generated synthetic sequences
        code_languages: language filter for code datasets
        skip_documents: number of docs to skip in FineWeb (for resume)
        byte_mode: if True, encode text as UTF-8 bytes (0-259) for BLT

    Yields:
        dict with 'input_ids' [seq_len], 'labels' [seq_len], 'source' str
    """
    from .streaming import text_to_bytes, EOS_BYTE

    # Initialize curated sources
    active_curated: list[_ActiveSource] = []
    failed_names = []

    for source in CURATED_SOURCES:
        try:
            # Probe: can we load at least one sample?
            test_stream = _load_hf_stream(source)
            next(test_stream)
            del test_stream
            active_curated.append(_ActiveSource(source, source.weight))
            print(f"  ✓ {source.name} ({source.hf_id})")
        except (TypeError, ValueError) as e:
            if "Couldn't cast" in str(e) or "Nested data" in str(e):
                # Schema cast error — retry with raw parquet fallback
                try:
                    test_stream = _load_parquet_fallback(source)
                    next(test_stream)
                    del test_stream
                    source._use_parquet_fallback = True
                    active_curated.append(_ActiveSource(source, source.weight))
                    print(f"  ✓ {source.name} ({source.hf_id}) [parquet fallback]")
                except Exception as e2:
                    failed_names.append(source.name)
                    print(f"  ✗ {source.name}: {e} (fallback also failed: {e2})")
            else:
                failed_names.append(source.name)
                print(f"  ✗ {source.name}: {e}")
        except Exception as e:
            failed_names.append(source.name)
            print(f"  ✗ {source.name}: {e}")

    if failed_names:
        print(f"  {len(failed_names)} sources unavailable: {', '.join(failed_names)}")
        print(f"  Weight redistributed to {len(active_curated)} active sources")

    # General language streams (Tier 3)
    fineweb_weight = TIER_WEIGHTS["general"] * 0.67   # 10% of total
    code_weight = TIER_WEIGHTS["general"] * 0.33      # 5% of total

    # Synthetic stream (Tier 6)
    synthetic_weight = TIER_WEIGHTS["synthetic"]
    if synthetic_data is None:
        synthetic_data = generate_all_synthetic()

    # Compute total weight for normalization
    curated_total = sum(s.weight for s in active_curated)
    total_weight = curated_total + fineweb_weight + code_weight + synthetic_weight

    # Normalize weights
    curated_weights = [(s, s.weight / total_weight) for s in active_curated]
    w_fineweb = fineweb_weight / total_weight
    w_code = code_weight / total_weight
    w_synthetic = synthetic_weight / total_weight

    # Start general streams
    fineweb_iter = stream_fineweb_edu(
        tokenizer, seq_len, skip_documents=skip_documents, byte_mode=byte_mode,
    )
    code_iter = _try_code_stream(tokenizer, seq_len, code_languages, byte_mode=byte_mode)

    # Synthetic stream
    synthetic_iter = _synthetic_text_stream(synthetic_data)

    # Encoding buffer (shared across all text sources)
    if byte_mode:
        eos_id = EOS_BYTE
    else:
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            eos_id = tokenizer.vocab_size - 1
    buffer: list[int] = []

    mode_label = "byte" if byte_mode else "token"
    print(f"  Curated stream ({mode_label}): {len(active_curated)} datasets + FineWeb + code + synthetic")

    while True:
        r = random.random()
        cumulative = 0.0
        text = None
        source_name = "unknown"

        # Check curated sources first
        for active, norm_weight in curated_weights:
            cumulative += norm_weight
            if r < cumulative:
                if active.alive:
                    text = active.next_text()
                    source_name = active.source.name
                    if text is None:
                        # Source exhausted, skip to fallback
                        active.alive = False
                break

        # FineWeb
        if text is None and r < cumulative + w_fineweb:
            cumulative += w_fineweb
            try:
                batch = next(fineweb_iter)
                batch['source'] = 'fineweb'
                yield batch
                continue
            except StopIteration:
                fineweb_iter = stream_fineweb_edu(
                    tokenizer, seq_len, byte_mode=byte_mode,
                )
                continue

        # Code
        if text is None and r < cumulative + w_fineweb + w_code:
            if code_iter is not None:
                try:
                    batch = next(code_iter)
                    batch['source'] = 'code'
                    yield batch
                    continue
                except StopIteration:
                    code_iter = _try_code_stream(
                        tokenizer, seq_len, code_languages, byte_mode=byte_mode,
                    )
                    continue

        # Synthetic
        if text is None:
            try:
                text = next(synthetic_iter)
                source_name = 'synthetic'
            except StopIteration:
                random.shuffle(synthetic_data)
                synthetic_iter = _synthetic_text_stream(synthetic_data)
                continue

        if text is None:
            continue

        # Encode and pack into sequences
        if byte_mode:
            encoded = text_to_bytes(text)
        else:
            encoded = tokenizer.encode(text, add_special_tokens=False)
        encoded.append(eos_id)
        buffer.extend(encoded)

        while len(buffer) >= seq_len + 1:
            chunk = buffer[:seq_len + 1]
            buffer = buffer[seq_len:]
            yield {
                "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                "labels": torch.tensor(chunk[1:], dtype=torch.long),
                "source": source_name,
            }


def _try_code_stream(tokenizer, seq_len, languages, byte_mode=False):
    """Try to start a code stream, return None on failure."""
    try:
        return stream_code(
            tokenizer, seq_len, languages=languages, byte_mode=byte_mode,
        )
    except Exception as e:
        print(f"  Code stream unavailable: {e}")
        return None


def _synthetic_text_stream(data: list[str]) -> Iterator[str]:
    """Yield text strings from pre-generated synthetic data."""
    for text in data:
        if text:
            yield text
