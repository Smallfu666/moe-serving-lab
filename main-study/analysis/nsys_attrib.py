"""Decode-step attribution from one nsys capture window (E4 prereg §5).

Reads the sqlite export of one `.nsys-rep` (kernel activity only; memcpy/memset are not counted) and writes one JSON:

- per device: the full decode-graph executions (kernels that share one graph launch's correlationId, with at least
  half the kernel count of the largest execution in the window), the steady region from the first to the last such
  execution's start, steps = executions - 1, and over that region the per-step wall, the per-class kernel-time sum,
  the union of all kernel intervals, idle = wall - union and overlap = sum - union. Idle is split into the gaps between
  executions and the gaps inside them.
- communication skew (TP/EP ranks): per step, the minimum communication time over devices is the transfer estimate
  and each device's excess over it is its wait.
- `--mode mixed`: the window is a replay window; time inside full decode executions is reported apart from the rest.

Classes are the first regex in CLASSES matching the lower-cased demangled kernel name; the rest is `other`.
"""

import argparse
import bisect
import json
import re
import sqlite3
import statistics as st
from collections import defaultdict

CLASSES = [
    ("comm", r"nccl|allreduce|all_reduce|cross_device_reduce|allgather|all_gather|reducescatter|reduce_scatter"
             r"|alltoall|all_to_all|lamport|moe_a2a|moecomm|moe_comm|mnnvl|twoshot|oneshot|custom_ar|mscclpp"),
    ("moe_gemm", r"fused_moe|moe_gemm|grouped_gemm|group_gemm|cutlass_moe|marlin_moe|bmm_|moe_mm|moefc|moe_fc"),
    ("moe_other", r"moe|expert|topk_softmax|topkgating|topk_sum|router|routing|permute|finalize|align_block"
                  r"|align_single_token|count_and_sort"),
    ("other_known", r"norm|rotary|rope_|_rope|sampling|topk_top_p|top_k_top_p|top_p|act_and_mul|silu|gelu|quant"),
    ("attention", r"flash::|flash_fwd|flashattn|fmha|attention|attn|batchdecode|batchprefill|prefillwithkv"
                  r"|decodewithkv|paged|merge_state|mergestate|store_kv|set_kv|kvcache|kv_cache"),
    ("dense_gemm", r"gemm|nvjet|cublas|cutlass|xmma|matmul|w8a8|fp8_blockwise|deep_gemm|gemv|splitk|wgmma|tcgen"),
]
CLASS_NAMES = [c for c, _ in CLASSES] + ["other"]
BASE_NAME_CLASSES = {"other_known"}
G_CLASS_SHARE = 0.03
_RX = [(c, re.compile(r)) for c, r in CLASSES]


def classify(name):
    # other_known matches on the base name: template arguments such as GemmType::Normal or QuantParams would
    # otherwise pull GEMMs into it; the other classes need the template text (FA3, FlashInfer attention).
    low, base = name.lower(), base_name(name).lower()
    for c, rx in _RX:
        if rx.search(base if c in BASE_NAME_CLASSES else low):
            return c
    return "other"


def base_name(name):
    out, depth = [], 0
    for ch in name:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return "".join(out).split("(")[0].strip()


def big_kernels(by_name, total):
    grouped = defaultdict(int)
    for n, v in by_name.items():
        grouped[(base_name(n), classify(n))] += v
    return [{"base_name": b[:160], "class": c, "share_of_sum": v / total}
            for (b, c), v in sorted(grouped.items(), key=lambda kv: -kv[1]) if total and v / total >= G_CLASS_SHARE]


def union_len(iv):
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(iv):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def load(path):
    db = sqlite3.connect(path)
    cols = {r[1] for r in db.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
    if not cols:
        raise SystemExit(3)
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    gcol = "graphNodeId" if "graphNodeId" in cols else "NULL"
    rows = db.execute(f"SELECT deviceId, start, end, correlationId, demangledName, {gcol} FROM CUPTI_ACTIVITY_KIND_KERNEL")
    per_dev = defaultdict(list)
    for dev, s, e, corr, nid, gnode in rows:
        per_dev[dev].append((s, e, corr, names.get(nid, str(nid)), gnode is not None))
    return per_dev


def executions(kernels):
    groups = defaultdict(list)
    for s, e, corr, _, in_graph in kernels:
        if in_graph:
            groups[corr].append((s, e))
    ex = [(min(s for s, _ in g), max(e for _, e in g), len(g)) for g in groups.values()]
    if not ex:
        return []
    biggest = max(n for _, _, n in ex)
    return sorted(x for x in ex if x[2] * 2 >= biggest)


def clip(kernels, lo, hi):
    out = []
    for s, e, corr, name, g in kernels:
        s2, e2 = max(s, lo), min(e, hi)
        if e2 > s2:
            out.append((s2, e2, corr, name, g))
    return out


def device_decode(kernels):
    ex = executions(kernels)
    if len(ex) < 2:
        return {"executions": len(ex)}
    lo, hi = ex[0][0], ex[-1][0]
    steps = len(ex) - 1
    ks = clip(kernels, lo, hi)
    sums = defaultdict(int)
    by_name = defaultdict(int)
    for s, e, _, name, _ in ks:
        sums[classify(name)] += e - s
        by_name[name] += e - s
    union = union_len([(s, e) for s, e, *_ in ks])
    inter = 0
    for (s0, e0, _), (s1, _, _) in zip(ex, ex[1:]):
        gap_ks = [(max(s, e0), min(e, s1)) for s, e, *_ in ks if min(e, s1) > max(s, e0)]
        inter += max(0, (s1 - e0)) - union_len(gap_ks) if s1 > e0 else 0
    wall = hi - lo
    idle = wall - union
    total = sum(sums.values())
    per = lambda x: x / steps / 1e3
    top = sorted(by_name.items(), key=lambda kv: -kv[1])[:15]
    return {
        "executions": len(ex), "steps": steps, "kernels_per_execution": st.median(n for *_, n in ex),
        "step_us_mean": per(wall), "step_us_median": st.median(b[0] - a[0] for a, b in zip(ex, ex[1:])) / 1e3,
        "class_us_per_step": {c: per(sums.get(c, 0)) for c in CLASS_NAMES},
        "kernel_sum_us_per_step": per(total), "union_us_per_step": per(union), "idle_us_per_step": per(idle),
        "idle_between_executions_us_per_step": per(inter), "idle_inside_executions_us_per_step": per(idle - inter),
        "overlap_us_per_step": per(total - union),
        "closure_us": per(total) - per(total - union) + per(idle) - per(wall),
        "top_kernels": [{"name": n[:160], "class": classify(n), "share_of_sum": v / total} for n, v in top],
        "kernels_ge_3pct": big_kernels(by_name, total),
        "_ex": ex, "_ks": ks,
    }


def comm_skew(devs):
    ref = min(devs, key=lambda d: d)
    ref_ex = devs[ref]["_ex"]
    per_dev_steps = {}
    for d, info in devs.items():
        ex = info["_ex"]
        comm = []
        for k in range(len(ref_ex) - 1):
            t = ref_ex[k][0]
            j = min(range(len(ex) - 1), key=lambda i: abs(ex[i][0] - t)) if len(ex) > 1 else None
            if j is None:
                comm.append(None)
                continue
            lo, hi = ex[j][0], ex[j + 1][0]
            comm.append(sum(min(e, hi) - max(s, lo) for s, e, _, name, _ in info["_ks"]
                            if classify(name) == "comm" and min(e, hi) > max(s, lo)))
        per_dev_steps[d] = comm
    n = len(ref_ex) - 1
    mins = [min(v[k] for v in per_dev_steps.values() if v[k] is not None) for k in range(n)]
    out = {"transfer_estimate_us_per_step": st.mean(mins) / 1e3 if mins else None, "wait_us_per_step": {}}
    for d, v in per_dev_steps.items():
        out["wait_us_per_step"][str(d)] = st.mean(v[k] - mins[k] for k in range(n) if v[k] is not None) / 1e3
    return out


def device_mixed(kernels):
    ex = executions(kernels)
    lo, hi = min(s for s, *_ in kernels), max(e for _, e, *_ in kernels)
    in_ex = union_len([(s, e) for s, e, _ in ex])
    sums_in, sums_out, by_name = defaultdict(int), defaultdict(int), defaultdict(int)
    starts = [s for s, _, _ in ex]
    for s, e, _, name, _ in kernels:
        i = bisect.bisect_right(starts, s) - 1
        inside = i >= 0 and e <= ex[i][1]
        (sums_in if inside else sums_out)[classify(name)] += e - s
        by_name[name] += e - s
    return {"window_us": (hi - lo) / 1e3, "decode_executions": len(ex), "decode_execution_span_us": in_ex / 1e3,
            "union_us": union_len([(s, e) for s, e, *_ in kernels]) / 1e3,
            "class_us_inside_decode": {c: sums_in.get(c, 0) / 1e3 for c in CLASS_NAMES},
            "class_us_outside_decode": {c: sums_out.get(c, 0) / 1e3 for c in CLASS_NAMES},
            "kernels_ge_3pct": big_kernels(by_name, sum(by_name.values()))}


def analyse(path, mode, expect_devices, expect_steps=None):
    per_dev = load(path)
    out = {"report": path, "mode": mode, "devices": sorted(per_dev), "expect_devices": expect_devices,
           "kernel_rows": {str(d): len(k) for d, k in per_dev.items()}}
    out["gate_devices"] = len(per_dev) == expect_devices
    if mode == "decode":
        devs = {d: device_decode(k) for d, k in per_dev.items()}
        ok = {d: v for d, v in devs.items() if "steps" in v}
        if len(ok) == len(devs) and len(ok) > 1:
            out["comm_skew"] = comm_skew(ok)
        for v in devs.values():
            v.pop("_ex", None)
            v.pop("_ks", None)
        out["per_device"] = {str(d): v for d, v in devs.items()}
        if expect_steps is not None:
            out["gate_steps"] = all(abs(v.get("executions", 0) - expect_steps) <= 4 for v in devs.values())
    else:
        out["per_device"] = {str(d): device_mixed(k) for d, k in per_dev.items()}
    out["gate_class_other"] = not any(k["class"] == "other" for v in out["per_device"].values()
                                      for k in v.get("kernels_ge_3pct", []))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sqlite")
    ap.add_argument("--mode", choices=["decode", "mixed"], required=True)
    ap.add_argument("--expect-devices", type=int, required=True)
    ap.add_argument("--expect-steps", type=int)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = analyse(a.sqlite, a.mode, a.expect_devices, a.expect_steps)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    raise SystemExit(4 if not res["gate_devices"] else 5 if res.get("gate_steps") is False else 0)


if __name__ == "__main__":
    main()
