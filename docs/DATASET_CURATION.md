# Memoria Dataset Curation Plan

> Compiled from exhaustive HuggingFace research. ~115 datasets evaluated, ~50 selected.
> **Status:** Fully implemented in `memoria/data/curated.py` + `memoria/data/formatters.py`.
> 32 streaming sources registered across 5 tiers. All stream from HF, zero local disk.
> Used automatically for pretrained backbone training (`--config lfm2` or `--config qwen`).
> Failed sources gracefully redistribute weight at startup.

## Design Principle

Memoria trains with next-token prediction (L_token). The cognitive state earns its keep
only when **remembering, revising, or chaining facts is load-bearing for prediction**.
The current mix (90% generic web text) doesn't reward persistent state.

Target: **40-50% of training data should genuinely require persistent state** to predict well.

All QA/task datasets are converted to continuous text (`context + question + answer`)
so L_token naturally rewards having correct beliefs in state when reaching the answer tokens.

---

## Proposed Training Mix

```
Tier 1 — State Essential:           25%
Tier 2 — State Helps Significantly: 20%
Tier 3 — General Language:          15%
Tier 4 — NVIDIA Reasoning:          15%
Tier 5 — Tool Calling / Agentic:    10%
Tier 6 — Enhanced Synthetic:         5%
Reserve / Tuning:                   10%
```

---

## Tier 1: STATE IS ESSENTIAL (30%)

Data where you literally cannot predict the answer without cross-context memory.

### BABILong
- **HF ID:** `RMT-team/babilong-train-5k-samples`
- **Size:** 300K samples, 7.4 GB (Parquet)
- **License:** Free
- **Mix weight:** 8%
- **Tasks:** 10 tasks (qa1-qa10) at context lengths 0K-32K tokens
- **Why:** Designed specifically for memory-augmented architectures. Needle-in-haystack with reasoning — fact chaining, counting, negation, coreference. The paper showed recurrent memory transformers massively outperform standard LLMs here.
- **Format:** `input` (context), `question`, `target` (answer)

### FIREBALL (D&D Gameplay)
- **HF ID:** `lara-martin/FIREBALL`
- **Size:** 153K turns, 25K sessions (JSONL)
- **License:** CC-BY-4.0
- **Mix weight:** 6%
- **Why:** Gold-standard entity state tracking over long sessions. Structured game state (HP, spells, effects, combat) evolves causally. Each turn has before/after state annotations.
- **Format:** JSONL with `combat_state_before`, `commands_norm`, `automation_results`, `combat_state_after`

### WikiFactDiff
- **HF ID:** `Orange/WikiFactDiff` (config: `20210104-20230227_legacy`)
- **Size:** 328K rows (Parquet)
- **License:** CC-BY-SA-4.0
- **Mix weight:** 5%
- **Why:** Temporal knowledge revision. Facts categorized as NEW/OBSOLETE/STATIC between Jan 2021 and Feb 2023. Subject-relation-object triples with natural language verbalizations. Directly trains belief revision — the model learns that facts change.
- **Format:** SRO triples with `update_prompt`, `generalization_prompts`, `objects` (with decision: new/obsolete/static)

### OpenPI 2.0
- **HF ID:** `abhinavk/openpi_v2`
- **Size:** 10K-100K
- **License:** CC-BY-4.0
- **Mix weight:** 4%
- **Why:** Entity attribute state changes through procedural text. Predict (entity, attribute, before-state, after-state) at each step. Current SOTA models are far from competent. Directly exercises the core Memoria loop: observe step → update entity states → predict downstream effects.

### LoCoMo
- **HF ID:** `insight/locomo` (or `Percena/locomo-mc10`)
- **Size:** 50 dialogues, 7.5K QA pairs, up to 35 sessions per conversation (~9K tokens avg)
- **License:** Free
- **Mix weight:** 4%
- **Why:** Multi-session conversation where information from session 3 is needed in session 28. Tests exactly whether persistent state can carry memory across sessions. Machine-human pipeline generates dialogues grounded on personas and temporal event graphs.

### ExploreToM
- **HF ID:** `facebook/ExploreToM`
- **Size:** 13.3K examples
- **License:** CC-BY-NC-4.0
- **Mix weight:** 3%
- **Why:** False belief tracking / Theory of Mind. Characters move objects while others are absent. Questions probe what each character *believes* about object locations. Program-guided adversarial generation yields +27 accuracy on ToMi benchmark.

---

## Tier 2: STATE HELPS SIGNIFICANTLY (25%)

Multi-hop, causal, fact verification — state provides a clear advantage.

### ProofWriter
- **HF ID:** `tasksource/proofwriter`
- **Size:** 845K examples
- **License:** Free
- **Mix weight:** 4%
- **Why:** Multi-step deductive chains (0-10 hops). Facts + rules → proof derivation. Variable depth exercises causal edge chaining at different lengths.
- **Format:** theory, question, answer (True/False/Unknown), full proof chain

### VitaminC
- **HF ID:** `tals/vitaminc`
- **Size:** 489K examples
- **License:** CC-BY-SA-3.0
- **Mix weight:** 3%
- **Why:** Contrastive fact verification from Wikipedia edit revisions. Near-identical evidence pairs where one word changes the verdict. Trains fine-grained sensitivity to factual changes — +10% on adversarial fact verification.

### FEVER
- **HF ID:** `fever/fever`
- **Size:** 311K train examples
- **License:** CC-BY-SA-3.0
- **Mix weight:** 3%
- **Why:** Largest fact verification dataset. SUPPORTED/REFUTED/NOT_ENOUGH_INFO against Wikipedia. Evidence retrieval + verification trains belief updating.

### Corr2Cause
- **HF ID:** `causal-nlp/corr2cause`
- **Size:** 208K examples
- **License:** Free
- **Mix weight:** 3%
- **Why:** Correlation vs causation discrimination at scale. Given correlational premises about 4-6 variables, judge whether a causal hypothesis follows. Trains the model to avoid spurious causal edges.

### 2WikiMultiHopQA
- **HF ID:** `xanhho/2WikiMultihopQA`
- **Size:** 192K examples
- **License:** Free
- **Mix weight:** 2%
- **Why:** Largest multi-hop QA dataset. Cross-document evidence chains with annotations. Comparison, inference, and compositional questions.

### HotpotQA
- **HF ID:** `hotpotqa/hotpot_qa`
- **Size:** 113K QA pairs
- **License:** CC-BY-SA-4.0
- **Mix weight:** 2%
- **Why:** Multi-hop QA with sentence-level evidence annotations. Bridge-entity and comparison questions force connecting information across paragraphs.

### TGQA (Temporal Graph QA)
- **HF ID:** `sxiong/TGQA`
- **Size:** 107K rows
- **License:** MIT
- **Mix weight:** 2%
- **Why:** Stories → explicit temporal graphs → CoT reasoning. The graph annotations directly parallel Memoria's edge structure. Covers sequencing, duration, frequency, simultaneity.

### Goal-Step WikiHow
- **HF ID:** `tasksource/goal-step-wikihow`
- **Size:** 1.4M examples
- **License:** MIT
- **Mix weight:** 2%
- **Why:** Goal decomposition + step ordering from wikiHow. Three tasks: goal prediction, step ordering, step selection. Directly trains Telos goal system.

### MultiWOZ 2.2
- **HF ID:** `pfb30/multi_woz_v22`
- **Size:** 10.4K dialogues, 52K turns
- **License:** Apache 2.0
- **Mix weight:** 2%
- **Why:** Dialogue state tracking with full belief state annotations. Slot-value pairs added, modified, retracted across turns in 6+ domains.

### e-CARE
- **HF ID:** `12ml/e-CARE`
- **Size:** 17K examples
- **License:** Free
- **Mix weight:** 1%
- **Why:** Causal reasoning with natural language explanations. Each sample has premise, question type (cause/effect), choices, and `conceptual_explanation` articulating why the causal link holds.

### OpenToM
- **HF ID:** `SeacowX/OpenToM`
- **Size:** 16K examples
- **License:** Free
- **Mix weight:** 1%
- **Why:** Longer narratives (200-490 words) with personality-driven characters. Questions probe physical and psychological mental states.

---

## Tier 3: GENERAL LANGUAGE (15%)

Reduced from 90% to 15%. Still needed for language fluency.

### FineWeb-Edu
- **HF ID:** `HuggingFaceFW/fineweb-edu` (sample-10BT)
- **Size:** 10B tokens (streaming)
- **License:** Free
- **Mix weight:** 10%

### StarCoderData
- **HF ID:** `bigcode/starcoderdata`
- **Size:** 250B tokens (streaming)
- **License:** Free
- **Mix weight:** 5%
- **Languages:** Python, JavaScript, Rust, Go

---

## Tier 4: NVIDIA REASONING (15%)

High-quality reasoning data with chain-of-thought, all CC-BY-4.0.

### OpenMathReasoning
- **HF ID:** `nvidia/OpenMathReasoning`
- **Size:** 5.5M solutions across 306K unique problems
- **License:** CC-BY-4.0
- **Mix weight:** 4%
- **Why:** 3.2M long CoT solutions + 566K GenSelect (selecting best among candidates). GenSelect is uniquely valuable — trains belief revision by evaluating multiple solution paths.

### Nemotron-CrossThink
- **HF ID:** `nvidia/Nemotron-CrossThink`
- **Size:** 10M-100M rows
- **License:** CC-BY-4.0
- **Mix weight:** 3%
- **Why:** Cross-domain RL reasoning across science, math, and general domains. Forces transfer of reasoning patterns — exercises belief generalization.

### Nemotron-MIND
- **HF ID:** `nvidia/Nemotron-MIND`
- **Size:** 100M-1B tokens
- **License:** CC-BY-4.0
- **Mix weight:** 2%
- **Why:** Math-Informed Synthetic Dialogues. 7 conversational styles over math content. Dialogues naturally exercise belief tracking.

### OpenCodeReasoning-2
- **HF ID:** `nvidia/OpenCodeReasoning-2`
- **Size:** 2.5M samples (1.4M Python + 1.1M C++)
- **License:** CC-BY-4.0
- **Mix weight:** 2%
- **Why:** Competitive programming + code critique. Code critique exercises error detection and belief revision about code correctness.

### Nemotron-Agentic-v1
- **HF ID:** `nvidia/Nemotron-Agentic-v1`
- **Size:** Substantial
- **License:** CC-BY-4.0
- **Mix weight:** 2%
- **Why:** Multi-turn agentic trajectories with tool decomposition. Requires persistent state tracking, belief updates based on tool outputs, and planning.

### OpenScience + OpenScienceReasoning-2
- **HF ID:** `nvidia/OpenScience` + `nvidia/OpenScienceReasoning-2`
- **Size:** 1M-10M combined
- **License:** CC-BY-4.0
- **Mix weight:** 1%
- **Why:** Multi-domain STEM reasoning with detailed traces.

### ChatQA2-Long-SFT-data
- **HF ID:** `nvidia/ChatQA2-Long-SFT-data`
- **Size:** 100K-1M (up to 131K tokens per example)
- **License:** CC-BY-NC-2.0
- **Mix weight:** 1%
- **Why:** Long-context SFT data specifically designed for long-range QA. Directly exercises persistent memory.

---

## Tier 5: TOOL CALLING / AGENTIC (10%)

Multi-step tool use is goal decomposition (Telos) + state tracking (tool outputs update beliefs)
+ causal chains (step N depends on step N-1's result). Perfect exercise for cognitive state.

### Nemotron-Agentic-v1
- **HF ID:** `nvidia/Nemotron-Agentic-v1`
- **Size:** 206K rows (5.8 GB), splits: `interactive_agent` + `tool_calling`
- **License:** CC-BY-4.0
- **Mix weight:** 3%
- **Why:** Best multi-turn agentic dataset available. Synthetic trajectories with three roles (user, agent, tool environment). Goal decomposition, tool selection, reasoning over outputs, multi-step completion. Filtered by LLM judge for quality.

### xLAM Function Calling 60K
- **HF ID:** `Salesforce/xlam-function-calling-60k`
- **Size:** 60K rows
- **License:** CC-BY-4.0
- **Mix weight:** 2%
- **Why:** Gold standard for single-turn function calling. 3,673 executable APIs across 21 categories. Triple-verified: format check → actual execution → semantic verification. >95% correctness. Tops Berkeley Function-Calling Leaderboard.

### ToolBench
- **HF ID:** `tuandunghcmut/toolbench-v1`
- **Size:** 126K rows, 469K API calls, 16,464 real RapidAPI endpoints
- **License:** Apache-2.0
- **Mix weight:** 2%
- **Why:** Massive scale with real-world APIs. Single-tool, intra-category multi-tool, and intra-collection multi-tool scenarios. Full DFSDT reasoning traces. ICLR 2024 spotlight.

### When2Call
- **HF ID:** `nvidia/When2Call`
- **Size:** 15K SFT + 9K preference rows
- **License:** CC-BY-4.0
- **Mix weight:** 1%
- **Why:** Uniquely focused on *decision-making*: when to call tools, when to ask follow-up, when to admit inability, edge cases. Includes DPO preference data. Addresses the critical "should I use a tool at all?" question.

### xLAM Irrelevance 7.5K
- **HF ID:** `MadeAgents/xlam-irrelevance-7.5k`
- **Size:** 7.5K rows
- **License:** CC-BY-4.0
- **Mix weight:** 1%
- **Why:** Negative examples — teaches when NOT to call tools. Built by removing the correct function from xlam-60k candidates. Counters the tendency to always make function calls. Essential complement.

### Hermes Function Calling v1
- **HF ID:** `NousResearch/hermes-function-calling-v1`
- **Size:** 10K-100K (multiple subsets)
- **License:** Apache-2.0
- **Mix weight:** 0.5%
- **Why:** Behind Hermes 2 Pro. Covers function calling, JSON structured output, agentic JSON mode. Subsets: `func-calling-singleturn`, `func-calling` (multi-turn), `json-mode-agentic`.

### ToolACE
- **HF ID:** `Team-ACE/ToolACE`
- **Size:** 11.3K rows
- **License:** Apache-2.0
- **Mix weight:** 0.5%
- **Why:** Self-evolution synthesis from 26,507 diverse APIs. Multi-agent dialog generation with formalized thinking. Dual-layer verification (rule + model). Quality over quantity.

### Tool-Use Multiturn Reasoning
- **HF ID:** `interstellarninja/tool-use-multiturn-reasoning`
- **Size:** 14.5K rows (26.6 MB)
- **License:** Apache-2.0
- **Mix weight:** 0.5%
- **Why:** Unique among tool calling datasets — system prompt encourages "extremely long chains of thought" before tool calls. Most FC datasets have short/no reasoning traces; this one has the **thinking visible**, giving L_token signal on reasoning tokens, not just call format. Multi-turn with realistic API schemas (Azure, LinkedIn, Netflix, etc.). High density for cognitive state training.

### Tool Calling Reserve (not in primary mix)

| Dataset | HF ID | Size | Notes |
|---------|--------|------|-------|
| Glaive FC v2 | `glaiveai/glaive-function-calling-v2` | 113K | Apache-2.0. Original large-scale FC dataset. Already included in Nemotron-Agentic. |
| ToolScale | `nvidia/ToolScale` | 4K | RL-focused orchestrator training. Small but for RL phase. |
| Nemotron RL Tool Use | `nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1` | 10K-100K | CC-BY-4.0. 838 tools. For RL fine-tuning phase. |
| Nemotron RL FC Pivot | `nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1` | 9.6K | CC-BY-4.0. Action-pivoted (each row = one tool-call decision point). Multi-domain (finance, utilities). For RL behavior cloning, not SFT. |
| FC ShareGPT | `hypervariance/function-calling-sharegpt` | 87K | Cleaned Glaive v2 in ShareGPT format. |
| APIGen MT 5K | `Salesforce/APIGen-MT-5k` | 5K | Research-only license (GPT-4 generated). High quality multi-turn. |
| APIBench | `gorilla-llm/APIBench` | 16.5K | Apache-2.0. 1,716 ML APIs from HF/Torch/TensorHub. |
| BFCL | `gorilla-llm/Berkeley-Function-Calling-Leaderboard` | 2K | Eval benchmark, not training. |
| Nemotron Workplace | `nvidia/Nemotron-RL-agent-workplace_assistant` | 1K-10K | 26 tools, 690 tasks. Small but focused. |
| Nemotron Calendar | `nvidia/Nemotron-RL-agent-calendar_scheduling` | 1K-10K | Constraint satisfaction. |

---

## Tier 6: ENHANCED SYNTHETIC (5%)

Scale up from 2,500 to ~50K+ sequences with harder, longer tasks.

### Existing Synthetic Generator
- **Source:** `memoria/data/synthetic.py`
- **Mix weight:** 3%
- **Enhancement needed:** More entity diversity, longer chains (10+ updates before query), cross-sequence dependencies, multi-source precision calibration

### bAbI Tasks
- **HF ID:** `facebook/babi_qa`
- **Size:** 10K-100K across 20 tasks
- **License:** CC-BY-3.0
- **Mix weight:** 1%
- **Why:** Classic 20 QA tasks. Complement BABILong's needle-in-haystack with the original structured reasoning tasks.

### CLUTRR
- **HF ID:** `CLUTRR/v1`
- **Size:** 16K-60K
- **License:** Free
- **Mix weight:** 1%
- **Why:** Compositional relational reasoning. Train on 2-3 hop relations, test on 2-10. Tests whether causal edge chaining generalizes to unseen depths.

---

## Reserve / Additional Candidates (10%)

Available for tuning the mix based on belief advantage measurements.

| Dataset | HF ID | Size | Why held in reserve |
|---------|--------|------|---------------------|
| ProLong 64K | `princeton-nlp/prolong-data-64K` | 31B tokens | Massive; for optional continued pretraining |
| NarrativeQA | `deepmind/narrativeqa` | 28.8K QA (41K tok/doc) | Book-length; may need special handling |
| Nemotron-Math-v2 | `nvidia/Nemotron-Math-v2` | 7M trajectories | Multiple solutions per problem for revision training |
| AceMath-RM-Data | `nvidia/AceMath-RM-Training-Data` | 2.1M examples | 6 responses/question = natural revision signal |
| SciFact | `allenai/scifact` | 1.4K claims | Small but high-quality evidence-based belief revision |
| Climate-FEVER | `tdiggelm/climate_fever` | 1.5K claims | Unique DISPUTED label (probabilistic beliefs) |
| HEALTH_FACT | `ImperialCollegeLondon/health_fact` | 12K | 4-class veracity with explanations |
| StrategyQA | `voidful/StrategyQA` | 2.8K | Implicit multi-step (model must infer reasoning steps) |
| CREAK | `amydeng2000/CREAK` | 11.5K | Entity knowledge + commonsense verification |
| MultiNLI | `nyu-mll/multi_nli` | 433K | NLI at scale (less state-dependent) |
| ANLI | `facebook/anli` | 169K | Adversarial NLI (hard contradiction detection) |
| TempReason | `tonytan48/TempReason` | 10K-100K | Temporal arithmetic and ordering |
| RecipeNLG | `mbien/recipe_nlg` | 2.2M | Procedural graphs (non-commercial license) |
| Nemotron-Math-Proofs | `nvidia/Nemotron-Math-Proofs-v1` | 580K proofs | Lean 4 formalizations + reasoning traces |
| Nemotron-ClimbMix | `nvidia/Nemotron-ClimbMix` | 100M-1B | High-quality pretraining (better scaling than alternatives) |
| SWE-bench | `princeton-nlp/SWE-bench` | 21.5K | Repository-level causal debugging |
| SGD | `google-research-datasets/schema_guided_dstc8` | 18K+ dialogues | Schema-aware goal pursuit |
| REALTALK | `scottgeng00/realtalk` | 10 participants, 21 days | Real-world long-term messaging (authentic but small) |
| Opus Reasoning 3000x | `crownelius/Opus-4.6-Reasoning-3000x` | 3K (use original, not filtered) | Claude-generated CoT with explicit `thinking` field. Dense but small, math/code only, self-contained (no state dependency). |

---

## Evaluation-Only Benchmarks (NOT in training mix)

These are held out to prove the thesis: "500M + experience > 10B."

| Benchmark | HF ID | What it measures |
|-----------|--------|------------------|
| **LongMemEval** | `xiaowu0162/longmemeval-cleaned` | Cross-session memory — THE benchmark for Memoria's value prop |
| **MemoryAgentBench** | `ai-hyz/MemoryAgentBench` | Test-time learning + long-range understanding (ICLR 2026) |
| **MemoryBench** | `THUIR/MemoryBench-Full` | Continual learning from interactions |
| **MuSiQue** | `dgslibisey/MuSiQue` | Anti-shortcut multi-hop (3x human-machine gap) |
| **NarrativeQA** | `deepmind/narrativeqa` | Book-length comprehension (41K tokens/doc) |
| **QuALITY** | `emozilla/quality` | Deep reading comprehension (can't skim to answer) |
| **InfiniteBench** | `xinrongzhang2022/InfiniteBench` | 100K+ token reasoning |
| **LongBench v2** | `THUDM/LongBench-v2` | Bilingual long-context (8K-2M words) |
| **L-Eval** | `L4NLP/LEval` | 20 sub-tasks, 3K-200K tokens per document |
| **BFCL** | `gorilla-llm/Berkeley-Function-Calling-Leaderboard` | Industry-standard function calling eval (single/multi/parallel) |

---

## Datasets Evaluated and Excluded

| Dataset | Why excluded |
|---------|-------------|
| CLadder (`causal-nlp/CLadder`) | Broken on HF — CSV schema mismatch. Re-evaluate if fixed. |
| OpenHermes / SlimOrca | Instruction-tuning data for SFT phase, not pretraining |
| ProPara (AllenAI) | Not on HF Hub (GitHub only). OpenPI 2.0 covers the same capability. |
| RecipeNLG | Non-commercial license |
| BookSum | Covered by NarrativeQA in reserve |
| Long-Data-Collections | Subset of other datasets already included |
| KILT | Meta-benchmark; individual components (TriviaQA, FEVER) already included |
| TriviaQA / Natural Questions | Knowledge QA but doesn't require persistent state |
| ELI5 | Long-form QA but solvable within-context |
| WikiHop/QAngaroo | Superseded by 2WikiMultiHopQA |
| Multi-News | Summarization; doesn't strongly exercise state |
| CL-Bench | Small, not on HF in usable format |
| TRACE | GitHub only, not on HF |
| TimeDial | Test-only (1.1K), too small for training |
| FOLIO | 1.2K examples — evaluation only |
| EntailmentBank | 1.8K trees — evaluation only |
| COPA (SuperGLUE) | 1K examples — evaluation only |
| WinoGrande | Commonsense, not state-dependent |
| OpenBookQA | Small, not state-dependent |
| ARC | Science QA, not state-dependent |
| NVIDIA safety/alignment datasets | Not relevant to cognitive state training |
| NVIDIA multimodal datasets | Memoria is text-only |
| NVIDIA embedding datasets | Different training objective |

---

## Backbone Model Recommendation

**Primary:** `LiquidAI/LFM2.5-350M`
- 350M params, hybrid conv+attention (10 LIV conv + 6 GQA), trained on 28T tokens
- Hidden dim 1024, 16 layers, 128K context
- License: LFM1.0 (free for research + commercial under $10M revenue)
- Rationale: Small enough that cognitive state must earn its keep. Best language capability per parameter at this scale. Hybrid arch complements cognitive state (conv handles local, state handles global).
- **Requires:** Adapter code for non-standard layer types (conv layers don't accept position_embeddings)

**Comparison baseline:** `Qwen/Qwen3-0.6B`
- Standard transformer, same tokenizer already in use
- Clean ablation target for architecture comparison

**Dropped:** `Qwen/Qwen3.5-2B-Base` — too capable, belief advantage will never go positive

---

## Additional Datasets (added post-curation)

User-supplied datasets integrated into the pipeline:

| Dataset | HF ID | Tier | Weight | Notes |
|---------|--------|------|--------|-------|
| xLAM FC 60K | `Salesforce/xlam-function-calling-60k` | Tool | 2% | User has gated access |
| Nemotron RL Conv Tool | `nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1` | Tool | 1.5% | Multi-turn agentic pivots |
| Nemotron RL FC | `nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1` | Tool | 1.5% | FC decision points |
| Nemotron RL SWE | `nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1` | Tool | 1% | Software eng trajectories |
| ToolMind | `Nanbeige/ToolMind` | Tool | 1% | split: open_datasets |
| GSM8K | `openai/gsm8k` | Reasoning | 2% | Grade school math |
| OpenMathInstruct-1 | `nvidia/OpenMathInstruct-1` | Reasoning | 2% | Math instruction |
| DeepCoder | `agentica-org/DeepCoder-Preview-Dataset` | Reasoning | 1.5% | config: codeforces |
| Codeforces | `open-r1/codeforces` | Reasoning | 1.5% | Competitive programming |
| High-Coder Multi-Turn | `Crownelius/High-Coder-Reasoning-Multi-Turn` | Reasoning | 1% | Multi-turn coding |
| Tiny-Codes | `nampdn-ai/tiny-codes` | General | 1% | Small code snippets |
| Tiny-WebText | `nampdn-ai/tiny-webtext` | General | 1% | Small web text |

---

## Unavailable Datasets

Script-based datasets deprecated by HuggingFace (no longer streamable):
- `lara-martin/FIREBALL`, `abhinavk/openpi_v2`, `fever/fever`
- `xanhho/2WikiMultihopQA`, `pfb30/multi_woz_v22`
- `facebook/babi_qa`, `CLUTRR/v1`

Their weight is automatically redistributed to active sources at startup.

---

## Implementation Notes

1. **Format conversion:** All QA datasets → `[context]\nQuestion: [q]\nAnswer: [a]` for NTP
2. **Streaming:** ALL datasets use HF streaming mode — zero local disk footprint
3. **Tokenizer:** Auto-detected from backbone (LFM2 65K vocab, Qwen3 151K vocab)
4. **Sequence packing:** Pack multiple short examples into seq_len chunks with EOS boundaries
5. **Graceful fallback:** Failed sources redistribute weight to remaining active sources
6. **Tool calling format:** Preserve function signatures, tool calls, responses as continuous text
7. **Multi-turn agentic:** Keep full trajectory as continuous text for NTP
8. **Code:** `memoria/data/curated.py` (registry + streaming), `memoria/data/formatters.py` (32 format functions)
