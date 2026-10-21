"""Validation, byte cross-check, load-imbalance estimate and the TP8+EP8 vs DP8+EP8 prediction table.

Inputs: results/aic_points.json (from run_aic.py) and, optionally, the E2 per-request replay records of one H8
(TP8+EP8) launch (`--replay-dir .../e2-h8fp8ep-l1`, files replay_c{c}.jsonl). Writes results/summary.json and prints
markdown tables. CPU only, standard library only.
"""

import argparse
import bisect
import json
from pathlib import Path

HERE = Path(__file__).parent

# E4 decode windows (nsys-attached), step time T (us; device mean, median within 1 %) per configuration and batch: main-study/results/e4-results.md
# "Per-token GPU time" table and the B = 1 rows of the independent E4 score (not in this repo). 16,384-token prompts.
MEASURED_STEP_US = {
    ("TP4", 1): 8301.8, ("TP4", 4): 11247.1, ("TP4", 8): 14172.3, ("TP4", 16): 16573.2,
    ("TP8+EP8", 1): 10030.8, ("TP8+EP8", 4): 11462.2, ("TP8+EP8", 8): 12927.6, ("TP8+EP8", 16): 15567.4,
}
# Attention class (sum basis) per GPU per step = G_attention x B / GPUs, same table.
MEASURED_ATTN_US = {
    ("TP4", 4): 1716.0, ("TP4", 8): 2300.4, ("TP4", 16): 3334.0,
    ("TP8+EP8", 4): 1693.2, ("TP8+EP8", 8): 2296.5, ("TP8+EP8", 16): 3344.0,
}
# From model/Qwen3-235B-A22B-Instruct-2507-FP8/config.json.
LAYERS, HIDDEN, NQ, NKV, HD = 94, 4096, 64, 4, 128
KV_BYTES = 2  # bf16 KV cache (--kv-cache-dtype bfloat16)
HBM_GBPS_SPEC = 4800.0  # H200 HBM3e vendor figure [SPEC], not measured
TP8_POOL_TOKENS = 2_127_459  # E2 H8-FP8 KV pool (e2-results.md)


def kv_bytes_per_token_per_gpu(cfg):
    heads = {"TP4": 1, "TP8+EP8": 1, "DP8+EP8": NKV}[cfg]  # ceil(4 / tp): TP8 replicates each KV head on 2 ranks
    return LAYERS * 2 * heads * HD * KV_BYTES


def attn_weight_bytes_per_gpu(cfg):
    tp = {"TP4": 4, "TP8+EP8": 8, "DP8+EP8": 1}[cfg]
    kv_heads = -(-NKV // tp)
    qkv = HIDDEN * (NQ // tp * HD + 2 * kv_heads * HD)
    o = (NQ // tp * HD) * HIDDEN
    return LAYERS * (qkv + o)  # fp8 weights, 1 byte each (scales ignored)


def dp8_pool_tokens():
    """KV pool if the TP8 KV budget is kept and the extra replicated attention weights come out of it."""
    kv_budget = TP8_POOL_TOKENS * kv_bytes_per_token_per_gpu("TP8+EP8")
    extra_w = attn_weight_bytes_per_gpu("DP8+EP8") - attn_weight_bytes_per_gpu("TP8+EP8")
    return int(8 * (kv_budget - extra_w) / kv_bytes_per_token_per_gpu("DP8+EP8"))


def load_points():
    pts = {}
    for p in json.loads((HERE / "results" / "aic_points.json").read_text()):
        pts[(p["config"], p["global_batch"], p["isl"])] = p
    return pts


def validation(pts):
    rows = []
    for (cfg, b), meas in MEASURED_STEP_US.items():
        p = pts[(cfg, b, 16480)]
        pred = p["tpot_ms"] * 1000
        attn_m = MEASURED_ATTN_US.get((cfg, b))
        attn_p = p["ops_ms_per_step"]["generation_attention"] * 1000
        kv_lb = b * 16480 * kv_bytes_per_token_per_gpu(cfg) / (HBM_GBPS_SPEC * 1e3)
        rows.append({"config": cfg, "B": b, "measured_us": meas, "aic_us": round(pred, 1),
                     "err_pct": round(100 * (pred / meas - 1), 1),
                     "attn_measured_us": attn_m, "attn_aic_us": round(attn_p, 1),
                     "attn_err_pct": round(100 * (attn_p / attn_m - 1), 1) if attn_m else None,
                     "kv_read_lower_bound_us": round(kv_lb, 1),
                     "kv_bw_frac_of_spec": round(kv_lb / attn_m, 3) if attn_m else None})
    ratios = []
    for b in (1, 4, 8, 16):
        m = MEASURED_STEP_US[("TP8+EP8", b)] / MEASURED_STEP_US[("TP4", b)]
        a = pts[("TP8+EP8", b, 16480)]["tpot_ms"] / pts[("TP4", b, 16480)]["tpot_ms"]
        ratios.append({"B": b, "measured_T8_over_T4": round(m, 3), "aic_T8_over_T4": round(a, 3),
                       "err_pct": round(100 * (a / m - 1), 1)})
    return rows, ratios


def replay_imbalance(replay_dir, ranks=8, dt=0.25):
    """Round-robin assignment of the recorded H8 requests to `ranks` attention ranks, sampled every dt seconds.

    Context of an in-flight request = its prompt tokens (generated tokens ignored). Returns, per replay point c,
    the time-weighted means of in-flight requests, mean context, and the attention-load imbalance
    I = max_rank(sum ctx) / mean_rank(sum ctx) over samples with >= 1 request in flight.
    """
    out = {}
    for f in sorted(Path(replay_dir).glob("replay_c*.jsonl"), key=lambda x: int(x.stem.split("_c")[1])):
        c = int(f.stem.split("_c")[1])
        reqs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        reqs = [r for r in reqs if r.get("error") is None and r.get("e2e_s") is not None]
        reqs.sort(key=lambda r: r["t_send"])
        rank_of = [i % ranks for i in range(len(reqs))]
        starts = [r["t_send"] for r in reqs]
        t0, t1 = starts[0], max(r["t_send"] + r["e2e_s"] for r in reqs)
        n_s = inflight_s = ctx_s = imb_s = bmax_s = 0.0
        t = t0
        while t < t1:
            hi = bisect.bisect_right(starts, t)
            load = [0] * ranks
            cnt = [0] * ranks
            n = 0
            for i in range(hi):
                r = reqs[i]
                if r["t_send"] + r["e2e_s"] > t:
                    load[rank_of[i]] += r["prompt_tokens"]
                    cnt[rank_of[i]] += 1
                    n += 1
            if n:
                n_s += 1
                inflight_s += n
                ctx_s += sum(load) / n
                imb_s += max(load) / (sum(load) / ranks)
                bmax_s += max(cnt) / (n / ranks)
            t += dt
        out[c] = {"mean_inflight": round(inflight_s / n_s, 2), "mean_ctx": round(ctx_s / n_s),
                  "attn_load_imbalance": round(imb_s / n_s, 2), "batch_imbalance": round(bmax_s / n_s, 2)}
    return out


def prediction(pts, imb):
    rows = []
    for b in (8, 16, 32, 64, 128):
        for isl in (8192, 16480, 32768):
            tp, dp = pts[("TP8+EP8", b, isl)], pts[("DP8+EP8", b, isl)]
            t_tp, t_dp = tp["tpot_ms"], dp["tpot_ms"]
            a_dp = dp["ops_ms_per_step"]["generation_attention"]

            def disp(pt):
                o = pt["ops_ms_per_step"]
                return o["generation_moe_pre_dispatch"] + o["generation_moe_post_dispatch"]

            t_dp_comm_as_tp = t_dp - (disp(dp) - disp(tp))
            t_mid = pts[("DP2xTP4+EP8", b, isl)]["tpot_ms"]
            row = {"B": b, "ctx": isl, "tp8ep8_ms": t_tp, "dp8ep8_ms": t_dp, "dp2tp4ep8_ms": t_mid,
                   "dp2tp4_over_tp": round(t_mid / t_tp, 3),
                   "dp_over_tp": round(t_dp / t_tp, 3),
                   "dp8_comm_as_tp_ms": round(t_dp_comm_as_tp, 3),
                   "dp8_comm_as_tp_over_tp": round(t_dp_comm_as_tp / t_tp, 3),
                   "tp8_fits_pool": b * (isl + 64) <= TP8_POOL_TOKENS,
                   "dp8_fits_pool": b * (isl + 64) <= dp8_pool_tokens()}
            near = min(imb, key=lambda c: abs(imb[c]["mean_inflight"] - b)) if imb else None
            if near is not None:
                i = imb[near]["attn_load_imbalance"]
                row["imbalance_used"] = i
                row["imbalance_from_c"] = near
                row["dp8ep8_imbalanced_ms"] = round(t_dp + (i - 1) * a_dp, 3)
                row["dp_imb_over_tp"] = round(row["dp8ep8_imbalanced_ms"] / t_tp, 3)
            rows.append(row)
    return rows


def md(rows, cols):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-dir", help="E2 H8 TP8+EP8 launch dir with replay_c*.jsonl (private raw data, not in this repo)")
    a = ap.parse_args()
    pts = load_points()
    val, ratios = validation(pts)
    imb = replay_imbalance(a.replay_dir) if a.replay_dir else {}
    pred = prediction(pts, imb)
    bytes_tab = [{"config": c, "kv_bytes_per_token_per_gpu": kv_bytes_per_token_per_gpu(c),
                  "attn_weight_GB_per_gpu": round(attn_weight_bytes_per_gpu(c) / 1e9, 2)}
                 for c in ("TP4", "TP8+EP8", "DP8+EP8")]
    summary = {"validation": val, "ratio_validation": ratios, "bytes": bytes_tab,
               "dp8_pool_tokens_est": dp8_pool_tokens(), "tp8_pool_tokens": TP8_POOL_TOKENS,
               "replay_imbalance": imb, "prediction": pred}
    (HERE / "results" / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(md(val, ["config", "B", "measured_us", "aic_us", "err_pct", "attn_measured_us", "attn_aic_us",
                   "attn_err_pct", "kv_read_lower_bound_us", "kv_bw_frac_of_spec"]))
    print()
    print(md(ratios, ["B", "measured_T8_over_T4", "aic_T8_over_T4", "err_pct"]))
    print()
    print(md(bytes_tab, ["config", "kv_bytes_per_token_per_gpu", "attn_weight_GB_per_gpu"]))
    print("DP8 pool est", dp8_pool_tokens(), "TP8 pool", TP8_POOL_TOKENS)
    print()
    print(md([{"c": c, **v} for c, v in imb.items()],
             ["c", "mean_inflight", "mean_ctx", "attn_load_imbalance", "batch_imbalance"]))
    print()
    print(md(pred, ["B", "ctx", "tp8ep8_ms", "dp8ep8_ms", "dp_over_tp", "dp8_comm_as_tp_over_tp", "imbalance_from_c", "imbalance_used",
                    "dp8ep8_imbalanced_ms", "dp_imb_over_tp", "dp2tp4ep8_ms", "dp2tp4_over_tp", "tp8_fits_pool",
                    "dp8_fits_pool"]))


if __name__ == "__main__":
    main()
