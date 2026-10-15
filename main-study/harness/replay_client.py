"""Closed-loop replay of tokenized agent sessions against SGLang's native /generate.

`concurrency` workers each take the next session from the slice and send its turns one after another, waiting for
each answer before sending the next turn; there is no think time. Turn k sends the recorded prompt
(`seq[:prompt_len]`) and asks for the recorded answer length, capped, with `ignore_eos`, so every arm receives
identical prompts and produces identical token counts.

One JSON line per request goes to --out; a summary line goes to --summary.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp


def load_sessions(path, start, count):
    with open(path) as f:
        lines = f.readlines()
    if start + count > len(lines):
        raise SystemExit(f"slice {start}+{count} exceeds {len(lines)} sessions")
    return [json.loads(line) for line in lines[start:start + count]]


async def one_request(http, url, ids, max_new, timeout_s):
    payload = {
        "input_ids": ids,
        "sampling_params": {"max_new_tokens": max_new, "ignore_eos": True, "temperature": 0.0},
        "stream": True,
    }
    t0 = time.perf_counter()
    t_first, meta, error = None, {}, None
    try:
        async with http.post(url + "/generate", json=payload, timeout=aiohttp.ClientTimeout(total=timeout_s)) as r:
            if r.status != 200:
                return {"error": f"http {r.status}: {(await r.text())[:200]}", "t_send": t0}
            async for raw in r.content:
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                body = line[5:].strip()
                if body == b"[DONE]":
                    break
                chunk = json.loads(body)
                if "error" in chunk:
                    error = str(chunk["error"])[:200]
                    break
                if t_first is None:
                    t_first = time.perf_counter()
                meta = chunk.get("meta_info", meta)
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        error = f"{type(e).__name__}: {e}"[:200]
    t_end = time.perf_counter()
    rec = {"t_send": t0, "ttft_s": None if t_first is None else t_first - t0, "e2e_s": t_end - t0, "error": error}
    for k in ("prompt_tokens", "completion_tokens", "cached_tokens", "finish_reason"):
        rec[k] = meta.get(k)
    n = rec["completion_tokens"] or 0
    rec["tpot_s"] = (rec["e2e_s"] - rec["ttft_s"]) / (n - 1) if rec["ttft_s"] is not None and n > 1 else None
    return rec


async def run(a):
    sessions = load_sessions(a.sessions, a.start, a.count)
    queue = asyncio.Queue()
    for s in sessions:
        queue.put_nowait(s)
    out = open(a.out, "w")
    errors = 0

    async def worker(wid, http):
        nonlocal errors
        while True:
            try:
                s = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            for k, turn in enumerate(s["turns"]):
                max_new = min(turn["answer_len"], a.max_answer)
                rec = await one_request(http, a.url, s["seq"][: turn["prompt_len"]], max_new, a.request_timeout)
                rec.update({"session": s["session"], "turn": k, "worker": wid, "max_new": max_new,
                            "replay_prompt_len": turn["prompt_len"]})
                if rec["error"] or rec["completion_tokens"] != max_new or rec["prompt_tokens"] != turn["prompt_len"]:
                    errors += 1
                out.write(json.dumps(rec) + "\n")
                if rec["error"]:
                    break

    t0 = time.perf_counter()
    conn = aiohttp.TCPConnector(limit=a.concurrency * 2)
    async with aiohttp.ClientSession(connector=conn) as http:
        await asyncio.gather(*(worker(i, http) for i in range(a.concurrency)))
    wall = time.perf_counter() - t0
    out.close()
    summary = {"concurrency": a.concurrency, "sessions": len(sessions), "start": a.start, "wall_s": wall,
               "bad_requests": errors}
    Path(a.summary).write_text(json.dumps(summary) + "\n")
    print(json.dumps(summary))
    return 0 if errors == 0 else 3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--sessions", required=True, help="output of load_trajectories.py")
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--max-answer", type=int, default=1024)
    ap.add_argument("--request-timeout", type=float, default=900)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    raise SystemExit(asyncio.run(run(ap.parse_args())))


if __name__ == "__main__":
    main()
