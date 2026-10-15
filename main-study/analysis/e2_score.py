"""E2 computations (definitions: e2-prereg.md §4–§5).

Library functions are pure and unit-tested; `main` reads launch directories written by run_arm.sh.
Usage: e2_score.py <config>=<arm-dir>[,<arm-dir>...] ...   -> JSON on stdout.
"""

import json
import sys
from pathlib import Path

from summarize_launch import max_running, phase_windows

TARGETS = (25.0, 50.0, 75.0, 100.0)
REGISTERED_TURNS = {1: 142, 4: 142, 8: 273, 16: 543, 32: 974, 64: 2021, 128: 4219}


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


def med(values):
    """Launch-level median: middle of three, mean of two (prereg §4); None with fewer than two."""
    v = sorted(x for x in values if x is not None)
    if len(v) < 2:
        return None
    return v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2


def rel_spread(values):
    v = [x for x in values if x is not None]
    m = med(v)
    return (max(v) - min(v)) / m if m else None


def launch_points(launch):
    """launch: {c: point-dict}. Returns {c: (interactivity, tput)} over complete points."""
    return {c: (p["interactivity"], p["tput_per_gpu"]) for c, p in launch.items() if p.get("complete")}


def aggregate(launches):
    """Per concurrency point: median point and spreads over the launches where the point is complete."""
    pts = [launch_points(x) for x in launches]
    out = {}
    for c in sorted({c for p in pts for c in p}):
        vals = [p[c] for p in pts if c in p]
        if len(vals) < 2:
            continue
        out[c] = {"point": (med([v[0] for v in vals]), med([v[1] for v in vals])),
                  "tput_spread": rel_spread([v[1] for v in vals]), "n": len(vals)}
    return out


def bracket(front, target):
    """The frontier points used for `target`: the exact match, or the two that bracket it; [] outside range."""
    for x, y in front:
        if x == target:
            return [(x, y)]
    for a, b in zip(front, front[1:]):
        if a[0] < target < b[0]:
            return [a, b]
    return []


def config_view(launches):
    """Median frontier, the spread of each median point, and per-launch frontiers."""
    agg = aggregate(launches)
    spread_of = {v["point"]: v["tput_spread"] for v in agg.values()}
    return {"front": frontier([v["point"] for v in agg.values()]), "spread_of": spread_of,
            "launch_fronts": [frontier(list(launch_points(x).values())) for x in launches]}


def union_view(view_a, view_b):
    """Platform view: union of two configurations' points; launch k = launch k of each."""
    fronts = [frontier(fa + fb) for fa, fb in zip(view_a["launch_fronts"], view_b["launch_fronts"])]
    return {"front": frontier(view_a["front"] + view_b["front"]),
            "spread_of": {**view_a["spread_of"], **view_b["spread_of"]}, "launch_fronts": fronts}


def matched_ratio(num, den, target, min_pairs=4):
    """Ratio of throughput per GPU at `target` with the §5 noise rule; num/den are config or platform views."""
    yn, yd = interpolate(num["front"], target), interpolate(den["front"], target)
    r = yn / yd if yn is not None and yd else None
    pn = [interpolate(f, target) for f in num["launch_fronts"]]
    pd = [interpolate(f, target) for f in den["launch_fronts"]]
    pairs = [a / b for a in pn if a is not None for b in pd if b]
    spreads = [num["spread_of"].get(p) for p in bracket(num["front"], target)] + \
              [den["spread_of"].get(p) for p in bracket(den["front"], target)]
    s = max(spreads) if spreads and all(x is not None for x in spreads) else None
    label = "NOT_EVALUABLE" if len(pairs) < min_pairs else separation(r, pairs, s)
    return {"target": target, "ratio": r, "num": yn, "den": yd, "pairs": pairs, "spread": s, "label": label}


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
        complete = (phase is not None and summ["turns_done"] == summ["turns_total"] == REGISTERED_TURNS.get(c)
                    and not summ["cut_by_deadline"])
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
