"""Estimate R16 throughput with session-sticky DP dispatch, from the measured round-robin launches. An estimate.

Per launch and concurrency c, with W the phase wall time, O its output tokens, N its new prefill tokens and R the
c = 1 prefill rate:
- the simulation gives the new-token ratio rho = prefill tokens (session_lb) / prefill tokens (round_robin) for the
  same phase shape (sessions 52 .. 52 + c);
- `coupling.py` gives t0 and t1, the TPOT p50 of requests whose decode saw < 0.1 and > 0.9 of its time under some
  prefill, and the measured mean overlap o;
- kappa = o / ((N / R) / W) turns summed prefill seconds into covered wall share.
Model: W = (O / c) * (t0 + (t1 - t0) * min(1, kappa * P / W)) + N / (c R), with P = N / R. The round-robin case
(rho = 1) is solved first and compared with the measured W; the sticky case uses rho * N. Rank collisions of the sticky
assignment are not modelled: the model is server-wide and splits prefill evenly over the c workers.
"""

import argparse
import json
import math
from pathlib import Path

from routing_sim import simulate
from workload import Sessions, launch_dir


def solve(ow, t0, t1, kappa, p, tw):
    b = ow * t0 + tw
    w = (b + math.sqrt(b * b + 4 * ow * (t1 - t0) * kappa * p)) / 2
    if kappa * p / w >= 1:
        w = ow * t1 + tw
    return w, min(1.0, kappa * p / w)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--coupling", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sessions = Sessions(a.replay)
    cal = json.loads(Path(a.calibration).read_text())
    coup = {(r["dir"], r["concurrency"]): r for r in json.loads(Path(a.coupling).read_text())}
    u = cal["R16-FP8"]
    rate = u["prefill_tokens_per_s"]
    rows = []
    for c in (4, 8, 16):
        idx = list(range(52, 52 + c))
        tab = {int(k): v for k, v in u["tpot_p50_s_by_c"].items()}
        pf = {pol: simulate(sessions, idx, c, 16, 539136, 64, rate, 0.0, tab[c], pol, 0)["prefill_tokens"]
              for pol in ("round_robin", "session_lb")}
        rho = pf["session_lb"] / pf["round_robin"]
        n4 = []
        for d in cal["N4-FP8"]["launches"]:
            recs = [json.loads(x) for x in open(launch_dir(a.raw, d) / f"replay_c{c}.jsonl")]
            wall = json.loads((launch_dir(a.raw, d) / f"replay_c{c}.summary.json").read_text())["wall_s"]
            n4.append(sum(r["completion_tokens"] for r in recs) / wall)
        n4_per_gpu = sum(n4) / len(n4) / 4
        for d in u["launches"]:
            recs = [json.loads(x) for x in open(launch_dir(a.raw, d) / f"replay_c{c}.jsonl")]
            wall = json.loads((launch_dir(a.raw, d) / f"replay_c{c}.summary.json").read_text())["wall_s"]
            out = sum(r["completion_tokens"] for r in recs)
            new = sum(r["prompt_tokens"] - r["cached_tokens"] for r in recs)
            cp = coup[(d, c)]
            t0, t1 = cp["tpot_p50_ms_overlap_lt_0.1"] / 1e3, cp["tpot_p50_ms_overlap_gt_0.9"] / 1e3
            kappa = cp["mean_overlap"] / ((new / rate) / wall)
            w_rr, o_rr = solve(out / c, t0, t1, kappa, new / rate, new / c / rate)
            w_st, o_st = solve(out / c, t0, t1, kappa, rho * new / rate, rho * new / c / rate)
            rows.append({"dir": d, "concurrency": c, "rho_new_tokens_session_over_rr": round(rho, 4),
                         "measured_wall_s": round(wall, 1), "model_wall_rr_s": round(w_rr, 1),
                         "model_rr_error_pct": round(100 * (w_rr - wall) / wall, 1),
                         "measured_out_tok_s": round(out / wall, 1),
                         "projected_sticky_out_tok_s": round(out / w_st, 1),
                         "projected_speedup_model_over_model": round(w_rr / w_st, 2),
                         "projected_speedup_over_measured": round(wall / w_st, 2),
                         "projected_sticky_tpot_ms": round(1e3 * (t0 + (t1 - t0) * o_st), 1),
                         "measured_rr_per_gpu_over_n4_per_gpu": round(out / wall / 16 / n4_per_gpu, 4),
                         "projected_sticky_per_gpu_over_n4_per_gpu": round(out / w_st / 16 / n4_per_gpu, 4)})
            print(json.dumps(rows[-1]), flush=True)
    Path(a.out).write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
