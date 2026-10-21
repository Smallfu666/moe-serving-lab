"""Service-time inputs for the routing simulation, fitted from logged requests.

Prefill: at c = 1 nothing queues, so TTFT = a + new_tokens / R with new_tokens = prompt_tokens - cached_tokens;
a and R come from least squares over every c = 1 request of the listed launches. Decode: TPOT p50 per concurrency
point, as logged by the client. Both are measured on one replica configuration and reused as constants.
"""

import argparse
import json
import statistics
from pathlib import Path

from workload import launch_dir


def fit(xs, ys):
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    res = [y - (a + b * x) for x, y in zip(xs, ys)]
    ss_tot = sum((y - my) ** 2 for y in ys)
    return a, b, 1 - sum(r * r for r in res) / ss_tot


def load(raw, dirs, conc):
    out = []
    for d in dirs:
        f = launch_dir(raw, d) / f"replay_c{conc}.jsonl"
        if f.exists():
            out += [json.loads(x) for x in open(f)]
    return [r for r in out if r["ttft_s"] is not None and r["cached_tokens"] is not None]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    units = {
        "G2-FP8": ["E2 G2-FP8 L1", "E2 G2-FP8 L2", "E2 G2-FP8 L3"],
        "R16-FP8": ["E3 R16-FP8 L1", "E3 R16-FP8 L2", "E3 R16-FP8 L3"],
        "N4-FP8": [f"E3 N4-FP8 replica{k}" for k in (1, 2, 3, 4)],
    }
    res = {}
    for name, dirs in units.items():
        recs = load(a.raw, dirs, 1)
        xs = [r["prompt_tokens"] - r["cached_tokens"] for r in recs]
        ys = [r["ttft_s"] for r in recs]
        icpt, slope, r2 = fit(xs, ys)
        tpot = {}
        for c in (1, 4, 8, 16, 32, 64):
            rs = [r["tpot_s"] for r in load(a.raw, dirs, c) if r["tpot_s"] is not None]
            if rs:
                tpot[c] = round(statistics.median(rs), 5)
        res[name] = {"launches": dirs, "c1_requests": len(recs), "ttft_intercept_s": round(icpt, 4),
                     "prefill_tokens_per_s": round(1 / slope, 1), "r2": round(r2, 4),
                     "new_tokens_range": [min(xs), max(xs)], "tpot_p50_s_by_c": tpot}
        print(name, json.dumps(res[name]))
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
