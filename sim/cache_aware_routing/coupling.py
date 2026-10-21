"""Does a request's decode slow down while other requests prefill? Client timestamps only, R16 and N4 launches.

For request i the prefill window is estimated as [t_first_i - new_i / R, t_first_i], with new_i = prompt - cached
from its record and R the c = 1 prefill rate from `results/calibration.json`. For request j, `overlap_j` is the share
of its decode window [t_first_j, t_end_j] covered by the union of the other requests' prefill windows. The script
fits TPOT_j = a + b * overlap_j by least squares per launch and concurrency, and reports TPOT for requests with
overlap < 0.1 and > 0.9. Prefill windows are estimates and overlap is a correlation, not an ablation.
"""

import argparse
import json
import statistics
from pathlib import Path

from calibrate import fit
from workload import launch_dir


def union_cover(lo, hi, windows):
    total, cur = 0.0, lo
    for a, b in windows:
        if b <= cur:
            continue
        if a >= hi:
            break
        a = max(a, cur)
        b = min(b, hi)
        if b > a:
            total += b - a
            cur = b
    return total / (hi - lo)


def analyse(recs, rate):
    ok = [r for r in recs if r["ttft_s"] is not None and r["tpot_s"] is not None and r["completion_tokens"] > 8]
    wins = []
    for r in ok:
        tf = r["t_send"] + r["ttft_s"]
        wins.append((tf - (r["prompt_tokens"] - r["cached_tokens"]) / rate, tf, id(r)))
    wins.sort()
    xs, ys = [], []
    for r in ok:
        lo, hi = r["t_send"] + r["ttft_s"], r["t_send"] + r["e2e_s"]
        others = [(a, b) for a, b, k in wins if k != id(r)]
        xs.append(union_cover(lo, hi, others))
        ys.append(r["tpot_s"])
    a, b, r2 = fit(xs, ys)
    low = [y for x, y in zip(xs, ys) if x < 0.1]
    high = [y for x, y in zip(xs, ys) if x > 0.9]
    return {"requests": len(ok), "mean_overlap": round(statistics.fmean(xs), 3),
            "tpot_fit_a_ms": round(1e3 * a, 2), "tpot_fit_b_ms": round(1e3 * b, 2), "r2": round(r2, 3),
            "tpot_p50_ms_overlap_lt_0.1": round(1e3 * statistics.median(low), 2) if low else None,
            "n_lt_0.1": len(low),
            "tpot_p50_ms_overlap_gt_0.9": round(1e3 * statistics.median(high), 2) if high else None,
            "n_gt_0.9": len(high)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cal = json.loads(Path(a.calibration).read_text())
    cases = [("R16-FP8", d, cal["R16-FP8"]["prefill_tokens_per_s"]) for d in cal["R16-FP8"]["launches"]]
    cases += [("N4-FP8", d, cal["N4-FP8"]["prefill_tokens_per_s"]) for d in cal["N4-FP8"]["launches"]]
    rows = []
    for unit, d, rate in cases:
        for c in (4, 8, 16):
            recs = [json.loads(x) for x in open(launch_dir(a.raw, d) / f"replay_c{c}.jsonl")]
            row = {"unit": unit, "dir": d, "concurrency": c, "rate_tok_s": rate, **analyse(recs, rate)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    Path(a.out).write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
