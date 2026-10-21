"""Parsing of the raw launch folders written by run_arm.sh: server logs, phase marks and replay records.

The raw folder (not in this repository) holds one directory per job: `<raw>/eN/<job>/<tag>/`; H100 jobs sit under
`<raw>/e2/h100/<job>/<tag>/`. Output of the scripts that use this module names configurations, job ids and launch
labels only, never paths or node names.
"""

import ast
import json
import re
from datetime import datetime
from pathlib import Path

TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(?: DP(\d+))?(?: TP(\d+))?[^\]]*\]")
PREFILL = re.compile(r"Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+)")
DECODE = re.compile(r"Decode batch, #running-req: (\d+), #token: (\d+).*cuda graph: (\w+), gen throughput \(token/s\): ([\d.]+)")
KV_ALLOC = re.compile(r"KV Cache is allocated\. dtype: (\S+), #tokens: (\d+), K size: ([\d.]+) GB, V size: ([\d.]+) GB")

# Directory-tag prefix -> configuration. Order matters: longer prefixes first. None = not data (VOID).
TAGS = [
    ("e1-gfp8-", "E1 G-FP8"), ("e1-gnv4-", "E1 G-NV4"), ("e1-hfp8tp4-", "E1 H-FP8-TP4"), ("e1-hfp8-", "E1 H-FP8"),
    ("e1-ref", "E1 REF (BF16)"),
    ("e2-g2fp8-", "G2-FP8"), ("e2-g2nv4-", "G2-NV4"), ("e2-h4fp8-", "H4-FP8"), ("e2-h8fp8ep-", "H8-FP8"),
    ("e2-h8fp8-", None), ("e2-h100x8ep-", "H100-8-FP8"), ("e2-h100-smoke", None),
    ("e3-n4fp8-", "N4-FP8"), ("e3-n4nv4-", "N4-NV4"), ("e3-r16fp8-", "R16-FP8"),
    ("e4-g2fp8", "E4 G2-FP8"), ("e4-g2nv4", "E4 G2-NV4"), ("e4-h4", "E4 H4-FP8"), ("e4-h8", "E4 H8-FP8"),
    ("e4-j1-30b-free", "E4 30B TP1 H200 (free)"), ("e4-j1-30b", "E4 30B TP1 H200"),
    ("e4-h-30b-free", "E4 30B TP2 H200 (free)"), ("e4-h-30b", "E4 30B TP2 H200"),
    ("e4-g2-30b-free", "E4 30B TP2 GB200 (free)"), ("e4-g2-30b", "E4 30B TP2 GB200"),
]
NOT_SCORED_JOBS = {"556631"}  # E4 J1 before erratum E4-E1; not scored (e4-prereg.md, erratum E4-E1)


def config_of(tag):
    for prefix, cfg in TAGS:
        if tag.startswith(prefix):
            return cfg
    raise ValueError(f"unknown launch tag {tag}")


def launch_label(tag):
    """Launch label without node names: l1/l2/l3 from the tag; N4 replicas are numbered later by the caller."""
    m = re.search(r"-(l\d)(?:-r(\d))?$", tag)
    if m:
        return m.group(1) + (f"-r{m.group(2)}" if m.group(2) else "")
    return ""


def launches(raw):
    """Yield (config, job, label, dir) for every launch directory under `raw`, sorted, VOID launches skipped.

    R16 yields one entry per launch with dir = the rank-0 directory; its peers are found by `r16_peers`.
    """
    raw = Path(raw)
    out = []
    for d in sorted(raw.glob("e*/**/server.log")):
        d = d.parent
        tag, job = d.name, d.parent.name
        cfg = config_of(tag)
        if cfg is None or job in NOT_SCORED_JOBS:
            continue
        label = launch_label(tag)
        if cfg == "R16-FP8" and not label.endswith("-r0"):
            continue
        out.append([cfg, job, label.removesuffix("-r0"), d])
    replica = {}
    for row in out:
        if row[0].startswith("N4-"):
            replica[(row[0], row[1])] = replica.get((row[0], row[1]), 0) + 1
            row[2] = f"replica{replica[(row[0], row[1])]}"
    return [tuple(r) for r in out]


def r16_peers(rank0_dir):
    d = Path(rank0_dir)
    stem = d.name.removesuffix("-r0")
    return [d.parent / f"{stem}-r{i}" for i in range(4)]


def read_lines(path):
    # newline="" keeps "\r" (progress bars) inside lines, so indices match the harness's phase_lines.txt.
    with open(path, errors="replace", newline="") as f:
        return f.read().split("\n")


def stamp(line):
    m = TS.match(line)
    if not m:
        return None, None
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"), (int(m.group(2)) if m.group(2) else 0)


def server_args(lines):
    for line in lines:
        if "server_args=" in line:
            return ast.literal_eval(line.split("server_args=", 1)[1].strip())
    return None


def phase_marks(arm_dir):
    """{step: {"start": line, "end": line}} from phase_lines.txt (line numbers index server.log split on '\\n')."""
    marks = {}
    p = Path(arm_dir) / "phase_lines.txt"
    for row in p.read_text().splitlines() if p.exists() else []:
        step, edge, n = row.split()
        marks.setdefault(step, {})[edge] = int(n)
    return marks


def replay_windows(arm_dir, lines):
    """{c: (t_first, t_last, cut)} wall-clock span of each replay phase, from the rank-0 server log."""
    status = (Path(arm_dir) / "arm_status.txt").read_text() if (Path(arm_dir) / "arm_status.txt").exists() else ""
    out = {}
    for step, m in phase_marks(arm_dir).items():
        if not step.startswith("replay:") or "start" not in m or "end" not in m:
            continue
        ts = [t for t, _ in map(stamp, lines[m["start"]:m["end"]]) if t]
        if ts:
            c = int(step.split(":")[1])
            cut = f"{step} wall_stop" in status or f"{step} rc=0" not in status
            out[c] = (ts[0], ts[-1], cut)
    return out


def events(lines, t0, t1):
    """Prefill and decode lines whose timestamp lies in [t0, t1]."""
    pf, dc = [], []
    for line in lines:
        t, dp = stamp(line)
        if t is None or t < t0 or t > t1:
            continue
        m = PREFILL.search(line)
        if m:
            pf.append({"t": t, "dp": dp, "new": int(m.group(2)), "cached": int(m.group(3))})
            continue
        m = DECODE.search(line)
        if m:
            dc.append({"t": t, "dp": dp, "running": int(m.group(1)), "graph": m.group(3) == "True",
                       "gen_tput": float(m.group(4))})
    return pf, dc


def replay_records(arm_dir, c):
    p = Path(arm_dir) / f"replay_c{c}.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else None


def pct(values, q):
    """Same percentile rule as summarize_launch.py / e2_score.py."""
    v = sorted(values)
    return v[min(len(v) - 1, int(len(v) * q))] if v else None
