"""Build the main-study summary figures from numbers quoted in main-study/results/*.md.

Every value below is copied by hand from the cited results line; nothing is computed from raw data.
Run: uv run --with matplotlib python main-study/figures/make_figures.py

Deferred, not drawn: per-launch ratio dots on fig1, and a union / per-GPU-step panel on fig3.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
TARGETS = [50, 75, 100]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "savefig.dpi": 160,
    }
)


def grouped_bars(ax, groups, series, labels, colors, hatches=None, fmt="{:.4f}", log=False):
    n = len(series)
    width = 0.8 / n
    for i, values in enumerate(series):
        xs = [g + (i - (n - 1) / 2) * width for g in range(len(groups))]
        label = labels[i]
        for x, v in zip(xs, values):
            if v is None:
                ax.text(x, 0.02 if log else 0.05, "n/e", ha="center", va="bottom", fontsize=7, color=MUTED,
                        transform=ax.get_xaxis_transform())
                continue
            ax.bar(x, v, width * 0.92, color=colors[i], edgecolor="white", linewidth=0.8,
                   hatch=hatches[i] if hatches else None, label=label)
            label = None
            ax.annotate(fmt.format(v), (x, v), xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    if log:
        ax.set_yscale("log")


def fig_e2():
    # e2-results.md:27, :29, :32
    ratios = {
        "G2-FP8 (TP2) / H4-FP8 (TP4)": [2.0080, 2.1141, 2.1756],
        "G2-NV4 (TP2) / H4-FP8 (TP4)": [3.3202, 3.9475, 4.3252],
        "H8-FP8 (TP8+EP8) / H4-FP8 (TP4)": [0.6158, 0.5347, 0.2827],
    }
    fig, ax = plt.subplots(figsize=(8, 4.2))
    grouped_bars(ax, [f"{t} tok/s/user" for t in TARGETS], list(ratios.values()), list(ratios.keys()), SERIES)
    ax.axhline(1.0, color=MUTED, linewidth=1, linestyle="--")
    ax.set_ylabel("throughput per GPU ratio (numerator / denominator)")
    ax.set_title("E2: per-GPU output throughput ratio at matched interactivity", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.text(0.01, 0.01, "Each platform at its own configuration; output tokens only; not a cost or GPU ranking. "
             "The uncontrolled H100 ratio is in figA1. Source: results/e2-results.md", fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "fig1_e2_per_gpu_ratios.png")
    plt.close(fig)


def fig_e2_h100_appendix():
    # e2-results.md:33
    ratios = {"H100-8-FP8 / H8-FP8 (both TP8+EP8)": [0.2599, 0.3005, None]}
    fig, ax = plt.subplots(figsize=(5, 3.6))
    grouped_bars(ax, [f"{t} tok/s/user" for t in TARGETS], list(ratios.values()), list(ratios.keys()),
                 [SERIES[3]], hatches=["////"])
    ax.axhline(1.0, color=MUTED, linewidth=1, linestyle="--")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("throughput per GPU ratio")
    ax.set_title("Appendix: H100 vs H200, NOT a controlled comparison", loc="left", color=INK, fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.text(0.01, 0.01, "Different cluster, driver (550 + CUDA-13 compatibility libraries) and container runtime.\n"
             "n/e = not evaluable (H100 frontier ends at 80.81). Source: results/e2-results.md", fontsize=6.5,
             color=MUTED)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "figA1_e2_h100_uncontrolled.png")
    plt.close(fig)


def fig_e3():
    # e2-results.md:51 (G2-FP8); e3-results.md:36 (N4-FP8), :38 (R16-FP8). The c of each frontier point is taken
    # from the independent score's per-point tables (not in this repository).
    concurrency = {
        "G2-FP8, 2 GPUs (TP2)": [16, 8, 4, 1],
        "N4-FP8, 4 GPUs (TP4+EP4)": [64, 32, 16, 8, 4, 1],
        "R16-FP8, 16 GPUs (TP16+attn-DP16+EP16, round-robin)": [16, 8, 4, 1],
    }
    frontiers = {
        "G2-FP8, 2 GPUs (TP2)": [(42.33, 272.39), (61.87, 186.04), (83.18, 129.34), (126.71, 60.99)],
        "N4-FP8, 4 GPUs (TP4+EP4)": [(26.11, 327.53), (39.30, 233.82), (57.66, 180.58), (77.14, 113.07),
                                     (97.72, 75.20), (138.74, 32.87)],
        "R16-FP8, 16 GPUs (TP16+attn-DP16+EP16, round-robin)": [(8.10, 7.16), (12.86, 5.32), (28.73, 4.76),
                                                                (94.94, 2.74)],
    }
    # e3-results.md:24, :25, :28
    ratios = {
        "N4-FP8 (TP4+EP4) / G2-FP8 (TP2)": [0.8503, 0.7974, 0.7078],
        "N4-NV4 (TP4+EP4) / G2-NV4 (TP2)": [0.7953, 0.8681, 0.7976],
        "R16-FP8 (4 nodes) / N4-FP8 (1 node)": [0.02028, 0.02781, None],
    }
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.2))
    # N4 labels go below-left, off its falling line.
    label_offset = [((3, 4), "left"), ((-4, -10), "right"), ((3, 4), "left")]
    for color, (offset, ha), (label, pts) in zip(SERIES, label_offset, frontiers.items()):
        xs, ys = zip(*pts)
        left.plot(xs, ys, "-o", color=color, linewidth=2, markersize=5, label=label)
        for (x, y), c in zip(pts, concurrency[label]):
            left.annotate(f"c={c}", (x, y), xytext=offset, textcoords="offset points", fontsize=6, color=color, ha=ha)
    left.annotate("R16: c = 32 cut by the 1 h end time,\nc = 64 and 128 never started", (8.10, 7.16),
                  xytext=(45, 14), fontsize=6.5, color=SERIES[2],
                  arrowprops={"arrowstyle": "-", "color": SERIES[2], "linewidth": 0.6})
    left.set_yscale("log")
    left.set_xlabel("interactivity (tok/s/user)")
    left.set_ylabel("output throughput (tok/s/GPU, log)")
    left.set_title("GB200 FP8 frontiers: G2, N4, R16", loc="left", color=INK)
    left.legend(frameon=False, fontsize=7)
    left.grid(True, axis="x", color=GRID, linewidth=0.6)

    grouped_bars(right, [f"{t} tok/s/user" for t in TARGETS], list(ratios.values()), list(ratios.keys()),
                 SERIES, fmt="{:.4g}", log=True)
    right.axhline(1.0, color=MUTED, linewidth=1, linestyle="--")
    right.set_ylabel("throughput per GPU ratio (log)")
    right.set_title("E3: registered per-GPU ratios", loc="left", color=INK)
    right.set_ylim(0.01, 6)
    right.legend(frameon=False, fontsize=7, loc="upper left", ncol=1)
    fig.text(0.01, 0.01, "Frontier points only: G2-FP8's c = 32 and 64 are dominated (KV-limited) and off its frontier. "
             "N4 vs G2 also differ in parallelism and node pool: not a GPU-count scaling result.\n"
             "R16 is one untuned configuration and its frontier covers only c ≤ 16; E3 does not isolate the cause. "
             "n/e = not evaluable. Sources: results/e2-results.md, e3-results.md", fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "fig2_e3_scaling.png")
    plt.close(fig)


def fig_e4():
    classes = ["comm", "moe_gemm", "moe_other", "other_known", "attention", "dense_gemm", "other", "idle",
               "−overlap"]
    # e4-results.md:89-94, share of ΔG = G(H4) − G(G2) per class at B = 4, 8, 16
    shares = {
        "G2-FP8 / H4-FP8": {
            4: [0.1372, 0.2800, 0.0453, 0.5839, 0.1592, 0.1760, 0.0374, 0.0644, -0.4834],
            8: [0.1211, 0.3942, 0.0063, 0.4369, 0.1632, 0.1149, 0.0299, 0.0455, -0.3121],
            16: [0.1168, 0.3546, -0.0095, 0.6363, 0.2068, 0.1051, 0.0290, 0.0575, -0.4966],
        },
        "G2-NV4 / H4-FP8": {
            4: [0.1123, 0.3830, 0.1010, 0.4861, 0.1293, 0.1525, 0.0308, 0.0553, -0.4503],
            8: [0.1002, 0.4803, 0.0594, 0.3648, 0.1343, 0.0984, 0.0245, 0.0392, -0.3011],
            16: [0.0931, 0.4723, 0.0457, 0.5061, 0.1625, 0.0855, 0.0226, 0.0463, -0.4341],
        },
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw={"width_ratios": [3, 3, 2]})
    for ax, (pair, by_b) in zip(axes[:2], shares.items()):
        n = len(by_b)
        width = 0.8 / n
        for i, (b, values) in enumerate(by_b.items()):
            xs = [c + (i - (n - 1) / 2) * width for c in range(len(classes))]
            ax.bar(xs, values, width * 0.92, color=SERIES[i], edgecolor="white", linewidth=0.6, label=f"B = {b}")
        ax.axhline(0.40, color=MUTED, linewidth=0.8, linestyle=":")
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.text(len(classes) - 0.5, 0.41, "F-DOM 0.40", ha="right", va="bottom", fontsize=6.5, color=MUTED)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=35, ha="right")
        ax.set_ylim(-0.6, 0.75)
        ax.set_ylabel("share of per-token GPU-time difference")
        flag = " (CLASSIFIER-GAP flag)" if pair.startswith("G2-FP8") else ""
        ax.set_title(f"E4 Q1 {pair}: no single dominant class{flag}", loc="left", color=INK, fontsize=9)
        ax.legend(frameon=False, fontsize=7, loc="upper left")

    # e4-results.md:161-163 (Q3, R16 vs N4 at c = 1–16, observational CPU log counts)
    q3 = axes[2]
    q3.bar(0, 0.96 - 0.95, 0.5, bottom=0.95, color=SERIES[0], label="N4-FP8 (4 GPUs)")
    q3.bar(1, 0.61 - 0.48, 0.5, bottom=0.48, color=SERIES[1], label="R16-FP8 (16 GPUs)")
    q3.annotate("0.95–0.96", (0, 0.96), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7)
    q3.annotate("0.48–0.61", (1, 0.61), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7)
    no_graph = [0, 0.125, 0.24, 0.38]
    xs = [2.6 + 0.35 * i for i in range(len(no_graph))]
    q3.bar(xs, no_graph, 0.3, color=SERIES[1], edgecolor="white", hatch="////")
    for x, v in zip(xs, no_graph):
        q3.annotate(f"{v:g}", (x, v), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7)
    q3.set_xticks([0.5, 3.125])
    q3.set_xticklabels(["prefill cached\nfraction (range,\nc = 1–16)", "R16 decode lines\nwithout CUDA graph\n(c rising →; N4: 0)"],
                       fontsize=7.5)
    q3.set_ylim(0, 1.08)
    q3.set_ylabel("fraction")
    q3.set_title("E4 Q3 R16 vs N4 (observational)", loc="left", color=INK)
    q3.legend(frameon=False, fontsize=7, loc="upper right")

    fig.text(0.01, 0.01, "Q1: profiled decode windows, B ∈ {4, 8, 16}, 16,384-token prompts, sum basis; shares "
             "add to >1 before −overlap. Descriptive, no class is called the cause. Q3: not preregistered, no "
             "counterfactual; co-occurs with R16's low ratio, cause not established. Source: results/e4-results.md",
             fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "fig3_e4_attribution_q3.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_e2()
    fig_e2_h100_appendix()
    fig_e3()
    fig_e4()
