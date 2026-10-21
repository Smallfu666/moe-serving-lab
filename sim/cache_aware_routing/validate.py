"""Replay the logged requests of a real launch through the cache model and compare cached tokens per request.

Each launch directory holds `phase_lines.txt` (phase order) and `replay_c*.jsonl` (one record per request with
t_send, ttft_s, e2e_s, prompt_tokens, cached_tokens). The harness calls `/flush_cache` before every phase
(`main-study/harness/run_arm.sh` flush_wait), so every replay phase starts from empty caches and the GSM8K and
logprob phases do not matter.

A request takes its pages at t_send + ttft_s (prefill has happened by the first token) or, with admit_at="send", at
t_send; it releases them at t_send + e2e_s. Only logged timing is used, never the logged cached_tokens. Requests are
dispatched across `replicas` caches round-robin in t_send order, SGLang's in-server DP default. After a flush every
cache is empty, so the round-robin starting rank only relabels caches and does not change the result.
"""

import argparse
import json
from pathlib import Path

from prefix_cache import PrefixCache
from workload import Sessions, request_keys, launch_dir

def phases(run_dir):
    out = []
    for line in (run_dir / "phase_lines.txt").read_text().splitlines():
        name, what = line.split()[0], line.split()[1]
        if what == "start":
            out.append(name)
    return out


def replay_records(run_dir, phase):
    conc = phase.split(":")[1]
    recs = [json.loads(x) for x in open(run_dir / f"replay_c{conc}.jsonl")]
    return sorted((r for r in recs if r["cached_tokens"] is not None), key=lambda r: r["t_send"])


def run(run_dir, sessions, capacity, page, replicas, admit_at="first"):
    rows, overflow, uid = [], 0, 0
    for ph in phases(run_dir):
        if ph.split(":")[0] != "replay":
            continue
        caches = [PrefixCache(capacity, page) for _ in range(replicas)]
        events = []
        for n, r in enumerate(replay_records(run_dir, ph)):
            i = sessions.index[r["session"]]
            keys = request_keys(sessions.keys(i, page), r["prompt_tokens"],
                                r["prompt_tokens"] + r["completion_tokens"], page, uid)
            uid += 1
            rep = n % replicas
            t_adm = r["t_send"] + (r["ttft_s"] if admit_at == "first" else 0.0)
            events.append((t_adm, 1, len(rows), rep, keys))
            events.append((r["t_send"] + r["e2e_s"], 0, len(rows), rep, keys))
            rows.append({"phase": ph, "session": r["session"], "turn": r["turn"], "replica": rep,
                         "prompt": r["prompt_tokens"], "logged": r["cached_tokens"], "sim": None})
        events.sort(key=lambda e: (e[0], e[1]))
        for t, is_admit, idx, rep, keys in events:
            if is_admit:
                rows[idx]["sim"] = min(caches[rep].admit(keys, t) * page, rows[idx]["prompt"])
            else:
                caches[rep].release(keys, t)
        overflow += sum(c.overflow_pages for c in caches)
    return rows, overflow


def summarize(rows):
    by = {}
    for r in rows:
        by.setdefault(r["phase"], []).append(r)
    out = []
    for ph, rs in by.items():
        p = sum(r["prompt"] for r in rs)
        meas = sum(r["logged"] for r in rs) / p
        sim = sum(r["sim"] for r in rs) / p
        exact = sum(r["sim"] == r["logged"] for r in rs) / len(rs)
        mae = sum(abs(r["sim"] - r["logged"]) for r in rs) / len(rs)
        out.append({"phase": ph, "requests": len(rs), "measured_hit": round(meas, 4), "sim_hit": round(sim, 4),
                    "error_pp": round(100 * (sim - meas), 2), "exact_request_share": round(exact, 4),
                    "mae_tokens_per_request": round(mae, 1)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", required=True, help="the frozen replay.jsonl")
    ap.add_argument("--raw", required=True, help="raw log folder; launch labels resolve through $LAUNCH_MAP")
    ap.add_argument("--config", required=True, help="validation_runs.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sessions = Sessions(a.replay)
    cfg = json.loads(Path(a.config).read_text())
    results = []
    for run_cfg in cfg["runs"]:
        for admit_at in ("first", "send"):
            rows, overflow = run(launch_dir(a.raw, run_cfg["dir"]), sessions, run_cfg["max_total_num_tokens"],
                                 run_cfg["page"], run_cfg["replicas"], admit_at)
            summ = summarize(rows)
            results.append({"dir": run_cfg["dir"], "label": run_cfg["label"], "admit_at": admit_at,
                            "overflow_pages": overflow, "phases": summ})
            print(run_cfg["label"], admit_at, overflow,
                  " ".join(f'{s["phase"]}:{s["measured_hit"]}/{s["sim_hit"]}' for s in summ), flush=True)
    Path(a.out).write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
