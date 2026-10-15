"""Per-turn replay prompts and the two prefix-hit upper bounds.

A session is a list of turns; each turn is (new_input_tokens, recorded_answer_tokens), already tokenized.
Turn k's prompt is every earlier turn's input and recorded answer, followed by turn k's input. Replaying the
recorded answer instead of the server's own output means the server's generated KV is never reused, so two bounds
are reported next to the measured hit rate:

- replay bound:  shared prefix of prompt k and prompt k-1, divided by len(prompt k)
- serving bound: (len(prompt k-1) + len(answer k-1)) / len(prompt k), what real serving could reuse

Both are undefined for the first turn of a session.
"""


def build_prompts(turns):
    prompts, history = [], []
    for new_input, answer in turns:
        prompt = history + list(new_input)
        prompts.append(prompt)
        history = prompt + list(answer)
    return prompts


def common_prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def turn_bounds(turns):
    prompts = build_prompts(turns)
    rows = []
    for k in range(1, len(prompts)):
        cur, prev = prompts[k], prompts[k - 1]
        prev_answer = len(turns[k - 1][1])
        rows.append({
            "turn": k,
            "prompt_tokens": len(cur),
            "replay_bound": common_prefix_len(prev, cur) / len(cur),
            "serving_bound": min(1.0, (len(prev) + prev_answer) / len(cur)),
        })
    return rows


def truncate_session(turns, max_context):
    """Drop the turns whose prompt plus answer would exceed max_context tokens, and everything after them."""
    kept, total = [], 0
    for new_input, answer in turns:
        if total + len(new_input) + len(answer) > max_context:
            break
        kept.append((new_input, answer))
        total += len(new_input) + len(answer)
    return kept
