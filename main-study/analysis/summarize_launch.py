"""Per-launch metrics from one arm directory written by run_arm.sh (definitions: prereg §4).

Usage: summarize_launch.py <arm-dir> [<arm-dir> ...]   -> one JSON object per launch on stdout.
"""

import json
import re
import statistics
import sys
from pathlib import Path


def pct(values, q):
    v = sorted(values)
    return v[min(len(v) - 1, int(len(v) * q))] if v else None


def replay_point(path):
    recs = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    summary = json.loads(path.with_name(path.stem + ".summary.json").read_text())
    ok = [r for r in recs if not r["error"]]
    ttft = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
    tpot = [r["tpot_s"] for r in ok if r["tpot_s"] is not None]
    out_tok = sum(r["completion_tokens"] or 0 for r in ok)
    prompt = sum(r["prompt_tokens"] or 0 for r in ok)
    cached = sum(r["cached_tokens"] or 0 for r in ok)
    bad = sum(1 for r in recs if r["error"] or r["completion_tokens"] != r["max_new"]
              or r["prompt_tokens"] != r["replay_prompt_len"])
    return {
        "requests": len(recs), "bad_requests": bad, "turns_done": summary.get("turns_done"),
        "cut_by_deadline": summary.get("cut_by_deadline"), "wall_s": summary["wall_s"],
        "ttft_p50_ms": 1e3 * pct(ttft, 0.5) if ttft else None, "ttft_p90_ms": 1e3 * pct(ttft, 0.9) if ttft else None,
        "tpot_p50_ms": 1e3 * statistics.median(tpot) if tpot else None,
        "out_tok_s": out_tok / summary["wall_s"] if summary["wall_s"] else None,
        "prefix_hit": cached / prompt if prompt else None,
    }


def max_running(lines):
    vals = [int(m) for line in lines for m in re.findall(r"#running-req: (\d+)", line)]
    return max(vals) if vals else None


def phase_windows(d):
    marks = {}
    pl = d / "phase_lines.txt"
    for line in pl.read_text().splitlines() if pl.exists() else []:
        step, edge, n = line.split()
        marks.setdefault(step, {})[edge] = int(n)
    return marks


def launch(arm_dir):
    d = Path(arm_dir)
    res = {"arm_dir": str(d), "arm_rc": (d / "arm_rc.txt").read_text().split() if (d / "arm_rc.txt").exists() else None}
    ready = d / "ready.txt"
    if ready.exists():
        res["ready"] = ready.read_text().split()
    log = (d / "server.log").read_text(errors="replace").splitlines() if (d / "server.log").exists() else []
    windows = phase_windows(d)
    for p in sorted(d.glob("replay_c*.jsonl")):
        res[p.stem] = replay_point(p)
        c = p.stem.removeprefix("replay_c")
        w = next((v for k, v in windows.items() if k.startswith(f"replay:{c}:") and "end" in v), None)
        res[p.stem]["max_running_req"] = max_running(log[w["start"]:w["end"]]) if w else None
    g = d / "gsm8k.summary.json"
    if g.exists():
        res["gsm8k"] = json.loads(g.read_text())
    return res


if __name__ == "__main__":
    for a in sys.argv[1:]:
        print(json.dumps(launch(a)))
