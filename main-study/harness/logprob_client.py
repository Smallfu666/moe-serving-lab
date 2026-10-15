"""Teacher-forced top-k logprobs on a fixed set of recorded agent turns.

Each item is one turn of a session from load_trajectories.py: the input is the recorded prompt plus the first
`forced` tokens of the recorded answer. The request starts logprobs one token before the answer, because SGLang
returns None for the first position of the window; the remaining `forced` positions are the answer tokens, each with
the logprob of the recorded token and the top-k alternatives. A response of any other shape is an error. Because the forced tokens come from the dataset, not from any arm,
arms (and a reference run on other hardware) can be compared position by position after the fact.
"""

import argparse
import asyncio
import json
from pathlib import Path

import aiohttp


def pick_turn(turns, max_prompt, forced):
    """The last turn whose prompt fits max_prompt and whose recorded answer has at least `forced` tokens."""
    best = None
    for k, t in enumerate(turns):
        if t["prompt_len"] <= max_prompt and t["answer_len"] >= forced:
            best = k
    return best


def build_items(sessions, n, max_prompt, forced):
    items = []
    for s in sessions:
        k = pick_turn(s["turns"], max_prompt, forced)
        if k is None:
            continue
        p = s["turns"][k]["prompt_len"]
        items.append({"session": s["session"], "turn": k, "prompt_len": p, "ids": s["seq"][: p + forced]})
        if len(items) == n:
            break
    if len(items) < n:
        raise SystemExit(f"only {len(items)} eligible turns, need {n}")
    return items


def check_shape(meta, forced, top_k):
    tok, top = meta.get("input_token_logprobs"), meta.get("input_top_logprobs")
    n = len(forced) + 1
    if not tok or not top or len(tok) != n or len(top) != n:
        return f"logprob shape: expected {n} entries"
    if top[0] is not None or tok[0][0] is not None:
        return "logprob shape: first entry is not None"
    if [t[1] for t in tok[1:]] != forced:
        return "logprob shape: token ids differ from the forced tokens"
    if any(len(x) != top_k for x in top[1:]):
        return f"logprob shape: top-k is not {top_k}"
    return None


async def run(a):
    lines = Path(a.sessions).read_text().splitlines()[a.start:]
    items = build_items((json.loads(x) for x in lines), a.n, a.max_prompt, a.forced)
    sem = asyncio.Semaphore(a.parallel)
    out = [None] * len(items)

    async def one(i, http):
        it = items[i]
        payload = {"input_ids": it["ids"], "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
                   "return_logprob": True, "logprob_start_len": it["prompt_len"] - 1, "top_logprobs_num": a.top_k}
        async with sem:
            try:
                async with http.post(a.url + "/generate", json=payload,
                                     timeout=aiohttp.ClientTimeout(total=a.request_timeout)) as r:
                    body = await r.json() if r.status == 200 else {"error": f"http {r.status}", "error_kind": "server"}
            except asyncio.TimeoutError:
                body = {"error": "TimeoutError", "error_kind": "client_timeout"}
            except aiohttp.ClientError as e:
                body = {"error": type(e).__name__, "error_kind": "client_exception"}
        meta = body.get("meta_info", {})
        forced = it["ids"][it["prompt_len"]:]
        if not body.get("error"):
            shape = check_shape(meta, forced, a.top_k)
            if shape:
                body = {"error": shape, "error_kind": "server"}
        out[i] = {"session": it["session"], "turn": it["turn"], "prompt_len": it["prompt_len"],
                  "forced_ids": forced,
                  "input_token_logprobs": meta.get("input_token_logprobs"),
                  "input_top_logprobs": meta.get("input_top_logprobs"), "error": body.get("error"),
                  "error_kind": body.get("error_kind")}

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=a.parallel)) as http:
        await asyncio.gather(*(one(i, http) for i in range(len(items))))
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    errors = sum(r["error"] is not None for r in out)
    print(json.dumps({"items": len(out), "errors": errors}))
    return 0 if errors == 0 else 3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--sessions", required=True)
    ap.add_argument("--start", type=int, required=True, help="first session line to draw items from")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--forced", type=int, default=64)
    ap.add_argument("--max-prompt", type=int, default=8192)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--parallel", type=int, default=32)
    ap.add_argument("--request-timeout", type=float, default=900)
    ap.add_argument("--out", required=True)
    raise SystemExit(asyncio.run(run(ap.parse_args())))


if __name__ == "__main__":
    main()
