"""Turn sampled agent trajectories into replay turns of token ids.

Input: the dataset's `trajectories.parquet` (each row carries its own tool schema in `tools`), a file of sampled
`trajectory_id`s, and the model's tokenizer directory. Output: one JSON line per session:

    {"session": id, "seq": [token ids of the longest kept prompt + its recorded answer],
     "turns": [{"prompt_len": n, "answer_len": m}, ...]}

Turn k's prompt is the chat template applied to every message before the k-th assistant message, with the generation
prompt appended; its recorded answer is what the template adds for that assistant message. Storing one shared
sequence per session only works if every prompt (plus its answer) is a prefix of the next prompt, so the loader
checks that; a session where it does not hold is left out and listed in `<out>.skipped`, so the kept sessions keep
the sampled order.
"""

import argparse
import hashlib
import json
from pathlib import Path

ROLE_FIELDS = {
    "system": ("role", "content"),
    "user": ("role", "content"),
    "assistant": ("role", "content", "tool_calls"),
    "tool": ("role", "content", "name", "tool_call_id"),
}


class NotPrefixStable(ValueError):
    pass


def clean_messages(trajectory):
    msgs = []
    for raw in trajectory:
        msg = {k: raw.get(k) for k in ROLE_FIELDS[raw["role"]]}
        if msg["role"] == "assistant":
            calls = msg.get("tool_calls")
            if calls:
                msg["tool_calls"] = [
                    {**c, "function": {**c["function"], "arguments": json.loads(c["function"]["arguments"])}}
                    if isinstance(c.get("function", {}).get("arguments"), str) else c
                    for c in calls
                ]
            else:
                msg.pop("tool_calls")
        msgs.append(msg)
    return msgs


def session_turns(msgs, encode, max_context, max_answer):
    """encode(messages, add_generation_prompt) -> token ids. Returns (seq, turns), truncated at max_context."""
    seq, turns = [], []
    for j, msg in enumerate(msgs):
        if msg["role"] != "assistant":
            continue
        prompt = encode(msgs[:j], True)
        full = encode(msgs[: j + 1], False)
        if full[: len(prompt)] != prompt:
            raise NotPrefixStable(f"assistant message {j}: answer does not extend its prompt")
        if prompt[: len(seq)] != seq:
            raise NotPrefixStable(f"assistant message {j}: prompt does not extend the previous turn")
        answer_len = len(full) - len(prompt)
        if len(prompt) + min(answer_len, max_answer) > max_context:
            break
        turns.append({"prompt_len": len(prompt), "answer_len": answer_len})
        seq = full
    return seq, turns


def drop_nulls(obj):
    """Parquet stores the union of all rows' struct fields, so keys a row never had come back as None."""
    if isinstance(obj, dict):
        return {k: drop_nulls(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [drop_nulls(v) for v in obj]
    return obj


def make_encoder(tokenizer, tools):
    def encode(messages, add_generation_prompt):
        ids = tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=add_generation_prompt, tokenize=True, return_dict=False
        )
        if not (isinstance(ids, list) and all(isinstance(i, int) for i in ids[:4])):
            raise TypeError(f"chat template returned {type(ids).__name__}, expected a flat list of token ids")
        return ids
    return encode


_WORKER = {}


def _init_worker(tokenizer_dir, max_context, max_answer):
    from transformers import AutoTokenizer
    _WORKER.update(tokenizer=AutoTokenizer.from_pretrained(tokenizer_dir), max_context=max_context,
                   max_answer=max_answer)


def _tokenize_one(job):
    tid, trajectory, tools = job
    encode = make_encoder(_WORKER["tokenizer"], drop_nulls(tools))
    try:
        seq, turns = session_turns(clean_messages(trajectory), encode, _WORKER["max_context"], _WORKER["max_answer"])
    except NotPrefixStable as e:
        return tid, None, str(e)
    if not turns:
        return tid, None, "no turn fits the context cap"
    return tid, json.dumps({"session": tid, "seq": seq, "turns": turns}, separators=(",", ":")), None


def _iter_rows(pf, wanted):
    """Yield (id, (trajectory, tools)) for wanted ids, one row group in memory at a time."""
    for rg in range(pf.num_row_groups):
        ids = pf.read_row_group(rg, columns=["trajectory_id"]).column(0).to_pylist()
        hit = [i for i, t in enumerate(ids) if t in wanted]
        if not hit:
            continue
        t = pf.read_row_group(rg, columns=["trajectory", "tools"]).take(hit)
        for j, i in enumerate(hit):
            yield ids[i], (t.column("trajectory")[j].as_py(), t.column("tools")[j].as_py())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--ids", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--max-context", type=int, default=32768)
    ap.add_argument("--max-answer", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import multiprocessing as mp

    import pyarrow.parquet as pq

    wanted = [line.strip() for line in a.ids.read_text().splitlines() if line.strip()]
    pf = pq.ParquetFile(a.parquet)
    jobs = dict(_iter_rows(pf, set(wanted)))
    missing = [t for t in wanted if t not in jobs]
    if missing:
        raise SystemExit(f"{len(missing)} sampled ids not in the parquet, first: {missing[0]}")
    ordered = ((tid, *jobs.pop(tid)) for tid in wanted)

    h, kept, skipped = hashlib.sha256(), 0, 0
    with mp.Pool(a.workers, _init_worker, (str(a.tokenizer), a.max_context, a.max_answer)) as pool, \
            a.out.open("w") as f, Path(str(a.out) + ".skipped").open("w") as fs:
        for tid, line, reason in pool.imap(_tokenize_one, ordered, chunksize=4):
            if line is None:
                skipped += 1
                fs.write(f"{tid}\t{reason}\n")
                continue
            kept += 1
            h.update((line + "\n").encode())
            f.write(line + "\n")
    print(f"sampled={len(wanted)} kept={kept} skipped={skipped} sha256={h.hexdigest()}")


if __name__ == "__main__":
    main()
