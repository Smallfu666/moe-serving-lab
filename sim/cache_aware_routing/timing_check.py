"""Check the routing simulation's service-time model against the launches it was calibrated on.

Runs `routing_sim.simulate` with the E2/E3 phase shape (sessions 52 .. 52 + c, c workers) for one G2-FP8 server and
for R16 under round-robin, and puts the simulated hit rate, TTFT p50 and output tokens/s next to the logged ones
(TTFT p50 over requests; output tokens/s = completion tokens / the phase's wall_s).
"""

import argparse
import json
import statistics
from pathlib import Path

from routing_sim import simulate
from workload import Sessions, launch_dir


def measured(raw, dirs, c):
    hits, ttft, tps = [], [], []
    for d in dirs:
        base = launch_dir(raw, d)
        recs = [json.loads(x) for x in open(base / f"replay_c{c}.jsonl")]
        summ = json.loads((base / f"replay_c{c}.summary.json").read_text())
        ok = [r for r in recs if r["ttft_s"] is not None]
        hits.append(sum(r["cached_tokens"] for r in ok) / sum(r["prompt_tokens"] for r in ok))
        ttft.append(statistics.median(r["ttft_s"] for r in ok))
        tps.append(sum(r["completion_tokens"] for r in ok) / summ["wall_s"])
        cut = summ["cut_by_deadline"]
    return {"hit": statistics.fmean(hits), "ttft_p50_s": statistics.fmean(ttft),
            "output_tok_per_s": statistics.fmean(tps), "launches": len(dirs), "cut_by_deadline": cut}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sessions = Sessions(a.replay)
    cal = json.loads(Path(a.calibration).read_text())
    cases = [("G2-FP8", 1, 514048, c) for c in (4, 8, 16, 32, 64)] + [("R16-FP8", 16, 539136, c) for c in (4, 8, 16, 32)]
    rows = []
    for unit, n, cap, c in cases:
        u = cal[unit]
        tab = {int(k): v for k, v in u["tpot_p50_s_by_c"].items()}
        sim = simulate(sessions, list(range(52, 52 + c)), c, n, cap, 64, u["prefill_tokens_per_s"],
                       max(u["ttft_intercept_s"], 0.0), tab[c], "round_robin", 0)
        meas = measured(a.raw, u["launches"], c)
        rows.append({"unit": unit, "concurrency": c, "measured": meas,
                     "sim": {k: sim[k] for k in ("hit", "ttft_p50_s", "output_tok_per_s")}})
        print(unit, c, json.dumps(rows[-1]["measured"]), json.dumps(rows[-1]["sim"]), flush=True)
    Path(a.out).write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
