"""Run NVIDIA AIConfigurator (aisimulate 0.12.0) static decode estimates for the configurations in this study.

Each point is `aiconfigurator cli estimate --estimate-mode static_gen`: a fixed batch decoding OSL=64 steps after an
ISL-token prompt, which is the shape of the E4 decode windows (16,384-token prompts, >= 96 warm tokens, 64 profiled
steps). `--batch-size` is per attention-DP rank in AIConfigurator, so DP8 points pass B/8.

Usage: python run_aic.py [--aic PATH_TO_aiconfigurator_BIN] [--out results/aic_points.json]
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# config.json of Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 @ e156cb4e (the E1-E4 revision), kept locally so runs do not
# depend on the hub.
MODEL = str(Path(__file__).with_name("model") / "Qwen3-235B-A22B-Instruct-2507-FP8")
OSL = 64
ISL_E4 = 16480

CONFIGS = {
    "TP4": dict(tp=4, dp=1, moe_tp=4, ep=1, gpus=4),
    "TP8+EP8": dict(tp=8, dp=1, moe_tp=1, ep=8, gpus=8),
    "DP8+EP8": dict(tp=1, dp=8, moe_tp=1, ep=8, gpus=8),
    "DP2xTP4+EP8": dict(tp=4, dp=2, moe_tp=1, ep=8, gpus=8),
}

VALIDATION = [(c, b, ISL_E4) for c in ("TP4", "TP8+EP8") for b in (1, 4, 8, 16)]
PREDICTION = [(c, b, isl) for c in ("TP8+EP8", "DP8+EP8", "DP2xTP4+EP8") for b in (8, 16, 32, 64, 128) for isl in (8192, 16480, 32768)]

OP_RE = re.compile(r"^\s+(generation_\w+)\s+([\d.]+) ms\s+([\d.]+) ms.*\[(\w+)\]\s*$")


def run_point(aic, system, cfg_name, batch, isl):
    c = CONFIGS[cfg_name]
    if batch % c["dp"]:
        raise ValueError(f"{cfg_name}: batch {batch} not divisible by dp {c['dp']}")
    cmd = [aic, "cli", "estimate", "--model-path", MODEL, "--system", system, "--backend", "sglang",
           "--estimate-mode", "static_gen", "--batch-size", str(batch // c["dp"]), "--isl", str(isl),
           "--osl", str(OSL), "--tp-size", str(c["tp"]), "--attention-dp-size", str(c["dp"]),
           "--moe-tp-size", str(c["moe_tp"]), "--moe-ep-size", str(c["ep"]),
           "--kvcache-quant-mode", "bfloat16", "--fmha-quant-mode", "bfloat16",
           "--no-color", "--detail", "time,source"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    out = p.stdout + p.stderr
    m = re.search(r"TPOT:\s+([\d.]+) ms", out)
    point = {"system": system, "config": cfg_name, "global_batch": batch, "per_rank_batch": batch // c["dp"],
             "isl": isl, "osl": OSL, "gpus": c["gpus"], "rc": p.returncode,
             "tpot_ms": float(m.group(1)) if m else None, "ops_ms_per_step": {}, "op_source": {}}
    ver = re.search(r"Backend:\s+sglang \(([^)]+)\)", out)
    point["db_version"] = ver.group(1) if ver else None
    mem = re.search(r"Memory \(GPU\):\s+([\d.]+) GB", out)
    point["mem_gb"] = float(mem.group(1)) if mem else None
    for line in out.splitlines():
        om = OP_RE.match(line)
        if om:
            point["ops_ms_per_step"][om.group(1)] = float(om.group(2)) / OSL
            point["op_source"][om.group(1)] = om.group(4)
    if m is None:
        point["error_tail"] = out.splitlines()[-15:]
    return point


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aic", default="aiconfigurator")
    ap.add_argument("--out", default=str(Path(__file__).with_name("results") / "aic_points.json"))
    a = ap.parse_args()
    pts = []
    for cfg, b, isl in VALIDATION + PREDICTION:
        pt = run_point(a.aic, "h200_sxm", cfg, b, isl)
        print(cfg, b, isl, pt["tpot_ms"], file=sys.stderr)
        pts.append(pt)
    Path(a.out).write_text(json.dumps(pts, indent=1) + "\n")


if __name__ == "__main__":
    main()
