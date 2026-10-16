"""Profiler-window clients for nsys capture (E4).

`decode`: hold a steady decode batch of B requests, each a real replay prompt cut to L tokens, then ask the server
to record `--steps` forward steps through `/start_profile` with the CUDA_PROFILER activity (nsys started with
`--capture-range=cudaProfilerApi`). The window is pure decode at batch B when every request has streamed at least
`--warm` tokens before the start and none has finished by the stop.

`trigger`: wait `--delay` seconds, then record `--steps` forward steps of whatever the server is doing (used beside a
replay client).

Both modes wait for the server log's "Stop profiling" line count to grow, and write one summary JSON.
Exit codes: 0 ok, 2 too few prompts of length L, 3 a request failed or finished before the window closed,
4 /start_profile refused, 5 the stop line never appeared, 6 server not idle afterwards.
"""

import argparse
import asyncio
import json
import time

import aiohttp

STOP_LINE = "Stop profiling"


def stop_count(path):
    with open(path, errors="replace", newline="") as f:
        return f.read().count(STOP_LINE)


def pick_prompts(path, start, batch, length, scan=2000):
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if i >= start + scan or len(out) == batch:
                break
            seq = json.loads(line)["seq"]
            if len(seq) >= length:
                out.append((i, seq[:length]))
    return out


class Stream:
    def __init__(self):
        self.tokens = 0
        self.done = False
        self.error = None


async def stream_one(http, url, ids, max_new, st, timeout_s):
    payload = {"input_ids": ids, "stream": True,
               "sampling_params": {"max_new_tokens": max_new, "ignore_eos": True, "temperature": 0.0}}
    try:
        async with http.post(url + "/generate", json=payload, timeout=aiohttp.ClientTimeout(total=timeout_s)) as r:
            if r.status != 200:
                st.error = f"http {r.status}"
                return
            async for raw in r.content:
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                body = line[5:].strip()
                if body == b"[DONE]":
                    break
                chunk = json.loads(body)
                if "error" in chunk:
                    st.error = str(chunk["error"])[:200]
                    return
                st.tokens = chunk.get("meta_info", {}).get("completion_tokens", st.tokens)
    except (asyncio.TimeoutError, aiohttp.ClientError) as e:
        st.error = f"{type(e).__name__}: {e}"[:200]
    finally:
        st.done = True


async def record_window(http, a, summary):
    before = stop_count(a.server_log)
    summary["t_profile_req"] = time.time()
    body = {"num_steps": a.steps, "activities": ["CUDA_PROFILER"], "profile_id": a.tag}
    async with http.post(a.url + "/start_profile", json=body, timeout=aiohttp.ClientTimeout(total=120)) as r:
        summary["start_profile_status"] = r.status
        summary["start_profile_body"] = (await r.text())[:300]
        if r.status != 200:
            return 4
    t0 = time.time()
    while stop_count(a.server_log) <= before:
        if time.time() - t0 > a.window_timeout:
            return 5
        await asyncio.sleep(0.2)
    summary["t_stop_seen"] = time.time()
    return 0


async def wait_idle(http, url, timeout_s):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            async with http.post(url + "/flush_cache", timeout=aiohttp.ClientTimeout(total=30)) as r:
                if r.status == 200:
                    return True
        except aiohttp.ClientError:
            pass
        await asyncio.sleep(2)
    return False


async def run_decode(a):
    summary = {"mode": "decode", "tag": a.tag, "batch": a.batch, "prompt_len": a.prompt_len, "steps": a.steps,
               "warm": a.warm, "max_new": a.max_new}
    prompts = pick_prompts(a.sessions, a.start, a.batch, a.prompt_len)
    summary["session_rows"] = [i for i, _ in prompts]
    if len(prompts) < a.batch:
        summary["rc"] = 2
        return summary
    async with aiohttp.ClientSession() as http:
        sts = [Stream() for _ in prompts]
        tasks = [asyncio.create_task(stream_one(http, a.url, ids, a.max_new, st, a.request_timeout))
                 for (_, ids), st in zip(prompts, sts)]
        t0 = time.time()
        while min(st.tokens for st in sts) < a.warm:
            if any(st.done for st in sts) or time.time() - t0 > a.warm_timeout:
                summary["rc"] = 3
                summary["errors"] = [st.error for st in sts]
                for t in tasks:
                    t.cancel()
                return summary
            await asyncio.sleep(0.05)
        summary["warm_s"] = time.time() - t0
        summary["tokens_at_start"] = [st.tokens for st in sts]
        rc = await record_window(http, a, summary)
        summary["tokens_at_stop"] = [st.tokens for st in sts]
        if rc == 0 and any(st.done for st in sts):
            rc = 3
        await asyncio.gather(*tasks, return_exceptions=True)
        summary["errors"] = [st.error for st in sts]
        if rc == 0 and any(summary["errors"]):
            rc = 3
        if rc == 0 and not await wait_idle(http, a.url, 120):
            rc = 6
    summary["rc"] = rc
    return summary


async def run_trigger(a):
    summary = {"mode": "trigger", "tag": a.tag, "steps": a.steps, "delay_s": a.delay}
    await asyncio.sleep(a.delay)
    async with aiohttp.ClientSession() as http:
        summary["rc"] = await record_window(http, a, summary)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["decode", "trigger"])
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--server-log", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--window-timeout", type=float, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sessions")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--prompt-len", type=int)
    ap.add_argument("--warm", type=int, default=96)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--warm-timeout", type=float, default=600)
    ap.add_argument("--request-timeout", type=float, default=1200)
    ap.add_argument("--delay", type=float, default=0)
    a = ap.parse_args()
    summary = asyncio.run(run_decode(a) if a.mode == "decode" else run_trigger(a))
    with open(a.out, "w") as f:
        json.dump(summary, f)
    raise SystemExit(summary["rc"])


if __name__ == "__main__":
    main()
