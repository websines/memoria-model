"""Format converters: raw HuggingFace samples → continuous text for NTP.

Each function takes a single sample dict (from a HF streaming dataset)
and returns a text string suitable for next-token prediction training.
Returns empty string if the sample is malformed (caller skips it).

Design: QA datasets become "context + question + answer" so that
L_token naturally rewards having correct beliefs when reaching answer tokens.
"""

import json


# ── Helpers ──

def _format_turns(turns: list[dict]) -> str:
    """Format conversation turns into continuous text.

    Handles both {'from': ..., 'value': ...} and {'role': ..., 'content': ...} formats.
    """
    parts = []
    for turn in turns:
        role = turn.get('from', turn.get('role', 'unknown'))
        content = turn.get('value', turn.get('content', ''))
        if content:
            parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _format_with_tools(sample: dict, turns_key: str = 'conversations',
                       tools_key: str = 'tools') -> str:
    """Format conversations with tool definitions as context."""
    parts = []
    tools = sample.get(tools_key)
    if tools:
        if isinstance(tools, str):
            parts.append(f"Available tools:\n{tools}")
        elif isinstance(tools, list):
            # Compact JSON for tool schemas
            parts.append(f"Available tools:\n{json.dumps(tools, separators=(',', ':'))}")
    turns = sample.get(turns_key, [])
    if isinstance(turns, list) and turns:
        parts.append(_format_turns(turns))
    return "\n".join(parts)


# ── Tier 1: State Essential ──

def format_babilong(sample: dict) -> str:
    """BABILong: context + question → answer. Tests needle-in-haystack memory."""
    context = sample.get('input', '')
    question = sample.get('question', '')
    answer = sample.get('target', '')
    if not question:
        return ''
    return f"{context}\nQuestion: {question}\nAnswer: {answer}"


def format_wikifactdiff(sample: dict) -> str:
    """WikiFactDiff: temporal knowledge revision. Facts change between time periods."""
    subject = sample.get('subject', '')
    relation = sample.get('relation', '')
    prompt = sample.get('update_prompt', '')
    objects = sample.get('objects', [])

    parts = []
    if prompt:
        parts.append(prompt)

    if isinstance(objects, list):
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            value = obj.get('object', '')
            decision = obj.get('decision', 'static')
            if decision == 'new':
                parts.append(f"New fact: {subject} {relation} {value}.")
            elif decision == 'obsolete':
                parts.append(f"No longer true: {subject} {relation} {value}.")
            else:
                parts.append(f"Still true: {subject} {relation} {value}.")

    return " ".join(parts) if parts else ''


def format_exploretom(sample: dict) -> str:
    """ExploreToM: false belief tracking / Theory of Mind."""
    story = sample.get('infilled_story', '')
    question = sample.get('question', '')
    answer = sample.get('expected_answer', '')
    if not story:
        return ''
    return f"{story}\nQuestion: {question}\nAnswer: {answer}"


# ── Tier 2: State Helps ──

def format_proofwriter(sample: dict) -> str:
    """ProofWriter: multi-step deductive reasoning with proof chains."""
    theory = sample.get('theory', '')
    question = sample.get('question', '')
    answer = sample.get('answer', '')
    proof = sample.get('allProofs', '')
    text = f"{theory}\nQuestion: {question}\nAnswer: {answer}"
    if proof:
        text += f"\nProof: {proof}"
    return text


def format_vitaminc(sample: dict) -> str:
    """VitaminC: contrastive fact verification from Wikipedia edits."""
    evidence = sample.get('evidence', '')
    claim = sample.get('claim', '')
    label = sample.get('label', -1)
    label_map = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT ENOUGH INFO"}
    verdict = label_map.get(label, str(label))
    if not evidence or not claim:
        return ''
    return f"Evidence: {evidence}\nClaim: {claim}\nVerdict: {verdict}"


def format_corr2cause(sample: dict) -> str:
    """Corr2Cause: correlation vs causation discrimination."""
    text = sample.get('input', '')
    label = sample.get('label', -1)
    label_map = {0: "No causal relationship", 1: "Causal relationship"}
    answer = label_map.get(label, str(label))
    if not text:
        return ''
    return f"{text}\nAnswer: {answer}"


def format_hotpotqa(sample: dict) -> str:
    """HotpotQA: multi-hop QA with cross-document evidence."""
    question = sample.get('question', '')
    answer = sample.get('answer', '')
    context = sample.get('context', {})
    if not question:
        return ''

    # Build context from title-sentences pairs
    context_parts = []
    titles = context.get('title', [])
    sentences = context.get('sentences', [])
    for title, sents in zip(titles, sentences):
        if isinstance(sents, list):
            context_parts.append(f"{title}: {''.join(sents)}")
        else:
            context_parts.append(f"{title}: {sents}")

    ctx_text = " ".join(context_parts)
    return f"{ctx_text}\nQuestion: {question}\nAnswer: {answer}"


def format_tgqa(sample: dict) -> str:
    """TGQA: temporal graph QA with story narratives."""
    story = sample.get('story', '')
    return story if story else ''


def format_goalstep(sample: dict) -> str:
    """Goal-Step WikiHow: goal decomposition and step ordering."""
    start = sample.get('startphrase', '')
    sent1 = sample.get('sent1', '')
    sent2 = sample.get('sent2', '')
    label = sample.get('label', 0)

    endings = [sample.get(f'ending{i}', '') for i in range(4)]
    answer = endings[label] if 0 <= label < len(endings) else ''

    if not start:
        return ''
    return (f"{start} {sent1} {sent2}\n"
            f"What happens next?\n"
            f"A: {endings[0]}\nB: {endings[1]}\nC: {endings[2]}\nD: {endings[3]}\n"
            f"Answer: {answer}")


def format_ecare(sample: dict) -> str:
    """e-CARE: causal reasoning with explanations."""
    premise = sample.get('premise', '')
    question = sample.get('question', '')
    choice1 = sample.get('choice1', '')
    choice2 = sample.get('choice2', '')
    label = sample.get('label', 0)
    explanation = sample.get('conceptual_explanation', '')

    answer = choice1 if label == 0 else choice2
    if not premise:
        return ''

    text = f"{premise}\nQuestion: {question}\nA: {choice1}\nB: {choice2}\nAnswer: {answer}"
    if explanation:
        text += f"\nExplanation: {explanation}"
    return text


def format_opentom(sample: dict) -> str:
    """OpenToM: long narrative Theory of Mind with personality-driven characters."""
    narrative = sample.get('narrative', sample.get('plot', ''))
    question = sample.get('question', '')
    if not narrative:
        return ''
    text = narrative
    if question:
        text += f"\nQuestion: {question}"
    return text


# ── Tier 4: NVIDIA Reasoning ──

def format_openmath(sample: dict) -> str:
    """OpenMathReasoning: chain-of-thought math solutions."""
    problem = sample.get('problem', '')
    solution = sample.get('generated_solution', '')
    if not problem:
        return ''
    return f"Problem: {problem}\nSolution: {solution}"


def format_crossthink(sample: dict) -> str:
    """Nemotron-CrossThink: cross-domain reasoning."""
    prompt = sample.get('prompt', '')
    return prompt if prompt else ''


def format_nemotron_mind(sample: dict) -> str:
    """Nemotron-MIND: math-informed synthetic dialogues."""
    text = sample.get('Text', '')
    return text if text else ''


# ── Tier 5: Tool Calling ──

def format_toolbench(sample: dict) -> str:
    """ToolBench: multi-tool API call trajectories."""
    return _format_with_tools(sample, turns_key='conversations', tools_key='tools')


def format_when2call(sample: dict) -> str:
    """When2Call: tool-use decision making."""
    return _format_with_tools(sample, turns_key='messages', tools_key='tools')


def format_xlam_irrelevance(sample: dict) -> str:
    """xLAM Irrelevance: negative examples — when NOT to call tools."""
    query = sample.get('query', '')
    tools = sample.get('tools', '')
    answers = sample.get('answers', '')
    if not query:
        return ''
    parts = [f"Query: {query}"]
    if tools:
        parts.append(f"Available tools: {tools}")
    if answers:
        parts.append(f"Answer: {answers}")
    return "\n".join(parts)


def format_hermes_fc(sample: dict) -> str:
    """Hermes Function Calling: multi-format tool use."""
    return _format_with_tools(sample, turns_key='conversations', tools_key='tools')


def format_toolace(sample: dict) -> str:
    """ToolACE: self-evolution tool use with formalized thinking."""
    system = sample.get('system', '')
    parts = []
    if system:
        parts.append(f"system: {system}")
    convs = sample.get('conversations', [])
    if convs:
        parts.append(_format_turns(convs))
    return "\n".join(parts)


def format_tool_multiturn(sample: dict) -> str:
    """Tool-Use Multiturn Reasoning: long chain-of-thought before tool calls."""
    return _format_with_tools(sample, turns_key='conversations', tools_key='tools')


# ── Additional datasets (user-supplied) ──

def format_xlam_fc(sample: dict) -> str:
    """xLAM Function Calling 60K: gold-standard single-turn function calling."""
    query = sample.get('query', '')
    tools = sample.get('tools', '')
    answers = sample.get('answers', '')
    if not query:
        return ''
    parts = [f"Query: {query}"]
    if tools:
        parts.append(f"Available tools: {tools}")
    if answers:
        parts.append(f"Answer: {answers}")
    return "\n".join(parts)


def format_nemotron_rl_agentic(sample: dict) -> str:
    """NVIDIA Nemotron RL Agentic: multi-turn agentic trajectories with tool pivots."""
    parts = []
    scenario = sample.get('scenario', sample.get('info', ''))
    if scenario:
        if isinstance(scenario, str):
            parts.append(scenario)
        elif isinstance(scenario, dict):
            parts.append(json.dumps(scenario, separators=(',', ':')))

    # responses_create_params contains the conversation turns
    responses = sample.get('responses_create_params', {})
    if isinstance(responses, dict):
        messages = responses.get('messages', [])
        if messages:
            parts.append(_format_turns(messages))

    expected = sample.get('expected_action', '')
    if expected:
        if isinstance(expected, str):
            parts.append(f"Expected action: {expected}")
        elif isinstance(expected, dict):
            parts.append(f"Expected action: {json.dumps(expected, separators=(',', ':'))}")

    return "\n".join(parts)


def format_nemotron_swe(sample: dict) -> str:
    """NVIDIA Nemotron SWE: software engineering agentic trajectories."""
    parts = []
    info = sample.get('info', '')
    if info:
        if isinstance(info, str):
            parts.append(info)
        elif isinstance(info, dict):
            parts.append(json.dumps(info, separators=(',', ':')))

    responses = sample.get('responses_create_params', {})
    if isinstance(responses, dict):
        messages = responses.get('messages', [])
        if messages:
            parts.append(_format_turns(messages))

    ref_msg = sample.get('ref_message', '')
    if ref_msg:
        parts.append(f"Reference: {ref_msg}")

    return "\n".join(parts)


def format_gsm8k(sample: dict) -> str:
    """GSM8K: grade school math with step-by-step solutions."""
    question = sample.get('question', '')
    answer = sample.get('answer', '')
    if not question:
        return ''
    return f"Problem: {question}\nSolution: {answer}"


def format_openmath_instruct(sample: dict) -> str:
    """OpenMathInstruct-1: math problems with generated solutions."""
    question = sample.get('question', '')
    solution = sample.get('generated_solution', '')
    if not question:
        return ''
    return f"Problem: {question}\nSolution: {solution}"


def format_deepcoder(sample: dict) -> str:
    """DeepCoder: competitive programming problems with solutions."""
    # Uses messages format
    messages = sample.get('messages', [])
    if messages:
        return _format_turns(messages)
    # Fallback to question/solution
    question = sample.get('question', sample.get('problem', ''))
    solution = sample.get('solution', sample.get('answer', ''))
    if question:
        return f"Problem: {question}\nSolution: {solution}"
    return ''


def format_toolmind(sample: dict) -> str:
    """ToolMind: tool-augmented reasoning."""
    return _format_with_tools(sample, turns_key='conversations', tools_key='tools')


def format_codeforces(sample: dict) -> str:
    """Open-R1 Codeforces: competitive programming problems."""
    name = sample.get('name', '')
    description = sample.get('description', '')
    input_spec = sample.get('input_specification', '')
    output_spec = sample.get('output_specification', '')
    solution = sample.get('solution', '')

    parts = []
    if name:
        parts.append(f"Problem: {name}")
    if description:
        parts.append(description)
    if input_spec:
        parts.append(f"Input: {input_spec}")
    if output_spec:
        parts.append(f"Output: {output_spec}")
    if solution:
        parts.append(f"Solution:\n{solution}")
    return "\n".join(parts) if parts else ''


def format_high_coder(sample: dict) -> str:
    """High-Coder Reasoning Multi-Turn: multi-turn coding with reasoning."""
    conversations = sample.get('conversations', [])
    if conversations:
        return _format_turns(conversations)
    return ''


def format_tiny_codes(sample: dict) -> str:
    """Tiny-Codes: small code snippets for training."""
    prompt = sample.get('prompt', '')
    response = sample.get('response', '')
    if not prompt:
        return ''
    return f"{prompt}\n{response}"


def format_tiny_webtext(sample: dict) -> str:
    """Tiny-WebText: small web text samples."""
    text = sample.get('textbook', sample.get('text', ''))
    return text if text else ''


# ── Code-heavy / Terminal Agent datasets ──

def format_nemotron_terminal(sample: dict) -> str:
    """Nemotron Terminal Corpus (cleaned): terminal coding agent SFT data.
    Teaches the model to use terminal commands, reason about outputs, chain actions.
    """
    # Conversation format with system/user/assistant turns
    conversations = sample.get('conversations', [])
    if conversations:
        return _format_turns(conversations)
    # Fallback: messages format
    messages = sample.get('messages', [])
    if messages:
        return _format_turns(messages)
    # Fallback: raw text
    return sample.get('text', sample.get('content', ''))


def format_swe_trajectories(sample: dict) -> str:
    """SWE-agent trajectories: full software engineering agent runs.
    Contains the complete trajectory of an agent solving real GitHub issues:
    reading files, running tests, writing patches.
    """
    # Typically has a trajectory field with the full agent run
    trajectory = sample.get('trajectory', '')
    if isinstance(trajectory, str) and trajectory:
        return trajectory
    if isinstance(trajectory, list):
        return "\n".join(str(t) for t in trajectory)

    # Fallback: messages/conversations
    messages = sample.get('messages', sample.get('conversations', []))
    if isinstance(messages, list) and messages:
        return _format_turns(messages)

    # Fallback: problem + patch
    parts = []
    problem = sample.get('problem_statement', sample.get('issue', ''))
    if problem:
        parts.append(f"Issue: {problem}")
    patch = sample.get('patch', sample.get('solution', ''))
    if patch:
        parts.append(f"Patch:\n{patch}")
    return "\n".join(parts) if parts else ''


def format_nemotron_opencode(sample: dict) -> str:
    """Nemotron SFT OpenCode: high-quality code instruction data.
    NVIDIA's curated code SFT dataset for training coding assistants.
    """
    messages = sample.get('messages', sample.get('conversations', []))
    if isinstance(messages, list) and messages:
        return _format_turns(messages)
    # Fallback
    prompt = sample.get('prompt', sample.get('input', sample.get('question', '')))
    response = sample.get('response', sample.get('output', sample.get('answer', '')))
    if prompt:
        return f"{prompt}\n{response}" if response else prompt
    return sample.get('text', '')


def format_opencode_reasoning(sample: dict) -> str:
    """OpenCodeReasoning-2: competitive programming + code critique.
    Python and C++ solutions with reasoning traces.
    """
    problem = sample.get('question', sample.get('problem', ''))
    solution = sample.get('solution', sample.get('generated_solution', ''))
    if not problem and not solution:
        # Messages format
        messages = sample.get('messages', [])
        if messages:
            return _format_turns(messages)
        return ''
    parts = []
    if problem:
        parts.append(f"Problem: {problem}")
    if solution:
        parts.append(f"Solution:\n{solution}")
    return "\n".join(parts)


def format_nemotron_agentic(sample: dict) -> str:
    """Nemotron-Agentic-v1: multi-turn agentic trajectories.
    Tool calling and interactive agent splits.
    """
    # Try conversations first
    conversations = sample.get('conversations', [])
    if isinstance(conversations, list) and conversations:
        parts = []
        tools = sample.get('tools', '')
        if tools:
            if isinstance(tools, str):
                parts.append(f"Available tools:\n{tools}")
            elif isinstance(tools, list):
                parts.append(f"Available tools:\n{json.dumps(tools, separators=(',', ':'))}")
        parts.append(_format_turns(conversations))
        return "\n".join(parts)

    # Messages format
    messages = sample.get('messages', [])
    if isinstance(messages, list) and messages:
        return _format_turns(messages)

    # Raw text fallback
    return sample.get('text', sample.get('content', ''))


def format_starcoderdata(sample: dict) -> str:
    """StarCoderData: raw code from GitHub repositories."""
    return sample.get('content', '')
