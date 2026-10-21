"""Markdown tables from results/*.json, for the README."""

import json
import statistics
from pathlib import Path

R = Path(__file__).parent / "results"
POLICIES = ["round_robin", "least_load", "session", "session_lb", "cache_aware", "cache_aware_abs4", "oracle"]


def load(name):
    return json.loads((R / name).read_text())


def validation():
    v = load("validation.json")
    print("### Validation: simulated vs logged prefix hit rate per phase (pages taken at first token)\n")
    print("| Launch | " + " | ".join(["phase: measured / simulated (error pp)"]) + " |")
    print("|---|---|")
    errs = {"first": [], "send": []}
    for run in v:
        errs[run["admit_at"]] += [abs(p["error_pp"]) for p in run["phases"]]
        if run["admit_at"] != "first":
            continue
        cells = [f'c{p["phase"].split(":")[1]}: {p["measured_hit"]:.4f} / {p["sim_hit"]:.4f} ({p["error_pp"]:+.2f})'
                 for p in run["phases"]]
        print(f'| {run["label"]} | ' + "; ".join(cells) + " |")
    for k, e in errs.items():
        print(f"\nPages taken at {'first token' if k == 'first' else 'send time'}: {len(e)} phase points, "
              f"max |error| {max(e):.2f} pp, mean |error| {statistics.fmean(e):.2f} pp.")


def routing():
    g = load("routing.json")
    rows = {}
    for r in g["grid"]:
        rows.setdefault((r["unit"], r["replicas"], r["concurrency"]), {})[r["policy"]] = r
    print("\n### Routing: prefix hit rate (mean of seeds; prefill tokens relative to round_robin in brackets)\n")
    print("| Replicas | C | " + " | ".join(POLICIES) + " |")
    print("|---|---|" + "---|" * len(POLICIES))
    for (unit, n, c), pol in rows.items():
        rr = pol["round_robin"]["prefill_tokens"]
        cells = [f'{pol[p]["hit"]:.3f} ({pol[p]["prefill_tokens"] / rr:.2f}x)' for p in POLICIES]
        print(f"| {n} x {unit} | {c} | " + " | ".join(cells) + " |")
    print("\n### Routing: load imbalance, max / mean over replicas "
          "(total prefill tokens computed; in-flight held tokens)\n")
    print("| Replicas | C | " + " | ".join(POLICIES) + " |")
    print("|---|---|" + "---|" * len(POLICIES))
    for (unit, n, c), pol in rows.items():
        cells = [f'{pol[p]["prefill_imbalance"]:.2f}; {pol[p]["inflight_tokens_max_over_mean"]:.2f}' for p in POLICIES]
        print(f"| {n} x {unit} | {c} | " + " | ".join(cells) + " |")
    print("\nUnbounded single-cache hit rate (upper bound): " + ", ".join(
        f'C={b["concurrency"]}: {b["unbounded_single_cache_hit"]:.4f}' for b in g["unbounded_bound"]) + ".")


def projection():
    p = load("r16_projection.json")
    print("\n### R16 with session-sticky DP dispatch: estimate (mean of 3 launches)\n")
    print("| c | new-token ratio | RR model wall error | measured out tok/s | projected | speed-up vs measured "
          "(model vs model) | "
          "projected TPOT ms | R16/N4 per GPU measured | projected |")
    print("|---|---|---|---|---|---|---|---|---|")
    for c in sorted({r["concurrency"] for r in p}):
        rs = [r for r in p if r["concurrency"] == c]
        m = lambda k: statistics.fmean(r[k] for r in rs)
        print(f'| {c} | {m("rho_new_tokens_session_over_rr"):.3f} | {m("model_rr_error_pct"):+.1f} % | '
              f'{m("measured_out_tok_s"):.1f} | {m("projected_sticky_out_tok_s"):.1f} | '
              f'{m("projected_speedup_over_measured"):.2f} ({m("projected_speedup_model_over_model"):.2f}) | '
              f'{m("projected_sticky_tpot_ms"):.1f} | {m("measured_rr_per_gpu_over_n4_per_gpu"):.4f} | '
              f'{m("projected_sticky_per_gpu_over_n4_per_gpu"):.4f} |')


def coupling():
    rows = load("coupling.json")
    print("\n### Decode under concurrent prefill (client timestamps; mean over launches)\n")
    print("| Config | c | mean overlap | TPOT p50 ms, overlap < 0.1 (n) | TPOT p50 ms, overlap > 0.9 (n) |")
    print("|---|---|---|---|---|")
    keys = sorted({(r["unit"], r["concurrency"]) for r in rows})
    for unit, c in keys:
        rs = [r for r in rows if (r["unit"], r["concurrency"]) == (unit, c)]
        lo = [r["tpot_p50_ms_overlap_lt_0.1"] for r in rs if r["tpot_p50_ms_overlap_lt_0.1"] is not None]
        hi = [r["tpot_p50_ms_overlap_gt_0.9"] for r in rs if r["tpot_p50_ms_overlap_gt_0.9"] is not None]
        print(f'| {unit} | {c} | {statistics.fmean(r["mean_overlap"] for r in rs):.3f} | '
              f'{statistics.fmean(lo):.2f} ({sum(r["n_lt_0.1"] for r in rs)}) | '
              + (f'{statistics.fmean(hi):.2f} ({sum(r["n_gt_0.9"] for r in rs)})' if hi else "— (0)") + " |")


def timing():
    rows = load("timing_check.json")
    print("\n### Service-time model check (phase shape of E2/E3; measured = mean of launches)\n")
    print("| Config | c | hit measured / sim | TTFT p50 s measured / sim | output tok/s measured / sim |")
    print("|---|---|---|---|---|")
    for r in rows:
        m, s = r["measured"], r["sim"]
        print(f'| {r["unit"]} | {r["concurrency"]} | {m["hit"]:.3f} / {s["hit"]:.3f} | '
              f'{m["ttft_p50_s"]:.3f} / {s["ttft_p50_s"]:.3f} | {m["output_tok_per_s"]:.1f} / {s["output_tok_per_s"]:.1f} |')


if __name__ == "__main__":
    validation()
    timing()
    routing()
    coupling()
    projection()
