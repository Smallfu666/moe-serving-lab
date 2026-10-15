"""E2 computations (definitions: e2-prereg.md §4–§5).

Library functions are pure and unit-tested; `main` reads launch directories written by run_arm.sh.
Usage: e2_score.py <config>=<arm-dir>[,<arm-dir>...] ...   -> JSON on stdout.
"""

import json
import sys
from pathlib import Path

from summarize_launch import max_running, phase_windows

TARGETS = (25.0, 50.0, 75.0, 100.0)


def pct(values, q):
    v = sorted(values)
    return v[min(len(v) - 1, int(len(v) * q))] if v else None


def request_rate(rec):
    """Per-request decode rate (output tokens - 1) / (end-to-end - TTFT); None if undefined."""
    n = rec.get("completion_tokens") or 0
    if rec.get("error") or rec.get("ttft_s") is None or n < 2:
        return None
    dt = rec["e2e_s"] - rec["ttft_s"]
    return (n - 1) / dt if dt > 0 else None


def mean_in_flight(recs):
    """Time-weighted mean number of requests in flight over the point's span."""
    iv = [(r["t_send"], r["t_send"] + r["e2e_s"]) for r in recs if r.get("e2e_s") is not None]
    if not iv:
        return None
    span = max(e for _, e in iv) - min(s for s, _ in iv)
    return sum(e - s for s, e in iv) / span if span > 0 else None


def point_metrics(recs, wall_s, n_gpus):
    ok = [r for r in recs if not r.get("error")]
    rates = [x for x in (request_rate(r) for r in ok) if x is not None]
    ttft = [r["ttft_s"] for r in ok if r.get("ttft_s") is not None]
    prompt = sum(r.get("prompt_tokens") or 0 for r in ok)
    return {
        "tput_per_gpu": sum(r.get("completion_tokens") or 0 for r in ok) / wall_s / n_gpus,
        "interactivity": pct(rates, 0.5),
        "ttft_p50_s": pct(ttft, 0.5),
        "ttft_p90_s": pct(ttft, 0.9),
        "prefix_hit": sum(r.get("cached_tokens") or 0 for r in ok) / prompt if prompt else None,
        "mean_in_flight": mean_in_flight(recs),
        "requests": len(recs),
    }


def frontier(points):
    """points: list of (interactivity, tput). Returns the non-dominated subset sorted by interactivity."""
    pts = [p for p in points if p[0] is not None and p[1] is not None]
    keep = []
    for i, (x, y) in enumerate(pts):
        dominated = any((x2 >= x and y2 >= y) and (x2 > x or y2 > y) for j, (x2, y2) in enumerate(pts) if j != i)
        if not dominated and (x, y) not in keep:
            keep.append((x, y))
    return sorted(keep)


def interpolate(front, target):
    """Throughput at `target` interactivity by linear interpolation on the frontier; None outside its range."""
    if not front:
        return None
    for x, y in front:
        if x == target:
            return y
    for (x0, y0), (x1, y1) in zip(front, front[1:]):
        if x0 < target < x1:
            return y0 + (y1 - y0) * (target - x0) / (x1 - x0)
    return None


def separation(ratio_med, pair_ratios, spread, floor=0.03, cap=0.15):
    """SEPARATED / NOT_SEPARATED / NOT_EVALUABLE for one matched-interactivity ratio."""
    if ratio_med is None or spread is None or spread > cap or any(r is None for r in pair_ratios):
        return "NOT_EVALUABLE"
    s = max(floor, 2 * spread)
    same_side = all(r > 1 for r in pair_ratios) or all(r < 1 for r in pair_ratios)
    return "SEPARATED" if abs(ratio_med - 1) >= s and same_side else "NOT_SEPARATED"


def rel_spread(values):
    v = [x for x in values if x is not None]
    if len(v) < 2:
        return None
    m = pct(v, 0.5)
    return (max(v) - min(v)) / m if m else None


def load_launch(arm_dir, n_gpus):
    d = Path(arm_dir)
    status = (d / "arm_status.txt").read_text().splitlines() if (d / "arm_status.txt").exists() else []
    done = {line.split()[1] for line in status if " rc=0 " in line}
    log = (d / "server.log").read_text(errors="replace").splitlines() if (d / "server.log").exists() else []
    windows = phase_windows(d)
    out = {}
    for p in sorted(d.glob("replay_c*.jsonl")):
        c = int(p.stem.removeprefix("replay_c"))
        summ = json.loads(p.with_name(p.stem + ".summary.json").read_text())
        phase = next((s for s in done if s.startswith(f"replay:{c}:")), None)
        complete = phase is not None and summ["turns_done"] == summ["turns_total"] and not summ["cut_by_deadline"]
        recs = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        w = windows.get(phase) if phase else None
        peak = max_running(log[w["start"]:w["end"]]) if w and "end" in w else None
        out[c] = {"complete": complete, "max_running_req": peak, **point_metrics(recs, summ["wall_s"], n_gpus)}
    return out


def main(argv):
    result = {}
    for spec in argv:
        name, dirs = spec.split("=", 1)
        n_gpus = int(name.split(":")[1]) if ":" in name else 1
        result[name] = [load_launch(x, n_gpus) for x in dirs.split(",")]
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
