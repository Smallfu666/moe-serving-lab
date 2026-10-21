"""E4 class times per GPU per decode step, re-expressed from the scored per-token table in results/e4-results.md.

Usage: python e4_per_gpu_step.py [path/to/e4-results.md]   -> Markdown tables on stdout.

The scored table gives G and each class in µs per output token, where G = step µs × GPUs ÷ B (device means, sum
basis). Multiplying by B ÷ GPUs gives µs per GPU per step, the time one device spends in that class during one
decode step. No new measurement: same numbers, different denominator.
"""

import sys
from pathlib import Path

GPUS = {"G2-FP8": 2, "G2-NV4": 2, "H4-FP8": 4, "H8-FP8": 8}
COLUMNS = ["config", "B", "step", "thr", "G", "comm", "moe_gemm", "moe_other", "other_known", "attention",
           "dense_gemm", "other", "idle", "overlap"]
SHOW = ["comm", "moe_gemm", "moe_other", "other_known", "attention", "dense_gemm", "other", "idle", "overlap"]
DEFAULT = Path(__file__).resolve().parents[1] / "results" / "e4-results.md"


def parse(path):
    text = Path(path).read_text().split("## Per-token GPU time", 1)[1].split("\n## ", 1)[0]
    rows = []
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != len(COLUMNS) or cells[0] not in GPUS:
            continue
        row = dict(zip(COLUMNS, cells))
        rows.append({"config": row["config"], "B": int(row["B"]),
                     **{k: float(row[k].replace("−", "-")) for k in COLUMNS[2:]}})
    return rows


def per_gpu_step(row):
    scale = row["B"] / GPUS[row["config"]]
    return {k: row[k] * scale for k in SHOW}


def main(path):
    rows = parse(path)
    by = {(r["config"], r["B"]): (r, per_gpu_step(r)) for r in rows}
    print("| Config | B | step µs | " + " | ".join(SHOW) + " |")
    print("|---|---|---|" + "---|" * len(SHOW))
    for r in rows:
        v = by[(r["config"], r["B"])][1]
        print(f"| {r['config']} | {r['B']} | {r['step']:.1f} | " + " | ".join(f"{v[k]:.1f}" for k in SHOW) + " |")
    print()
    print("H8-FP8 ÷ H4-FP8, per GPU per step (1.00 = unchanged from TP4 to TP8 + EP8; 0.50 = halved):")
    print()
    print("| B | step | " + " | ".join(SHOW) + " |")
    print("|---|---|" + "---|" * len(SHOW))
    for b in sorted({r["B"] for r in rows}):
        if ("H8-FP8", b) not in by or ("H4-FP8", b) not in by:
            continue
        (r8, v8), (r4, v4) = by[("H8-FP8", b)], by[("H4-FP8", b)]
        cells = [f"{v8[k] / v4[k]:.2f}" if v4[k] else "—" for k in SHOW]
        print(f"| {b} | {r8['step'] / r4['step']:.2f} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
