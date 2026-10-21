"""Server flags and resolved settings of every launch, grouped by configuration, from the raw launch folders.

Usage: python server_flags.py <raw-root>   -> Markdown tables on stdout.

"Passed" is the recorded command (`server_cmd.txt`, written by the harness from E2's H8 relaunch on) with the model path,
host, port, log level and node-rank arguments removed; launches without that file show the flags run_arm.sh builds
(`--tp <tp> --kv-cache-dtype bfloat16`) plus any non-default `--ep-size`, `--attn-dp-size` or `--moe-a2a-backend`, read back from the logged `server_args`
(the MoE runner is not reconstructed: on GB200 `auto` resolves to `flashinfer_trtllm` in the log).
Harness commit = `harness=` in the job's `job_info.txt` (a commit of this repository).
"Resolved" is what the server printed in `server_args` after its own defaults were applied.
KV heads per GPU = K bytes per token per GPU ÷ (layers × head_dim × 2 bytes), from the `KV Cache is allocated` line
(SGLang prints GiB as "GB"); layers 94 and head_dim 128 are the Qwen3-235B-A22B config values.
"""

import re
import sys
from collections import OrderedDict
from pathlib import Path

from serverlog import KV_ALLOC, launches, read_lines, server_args

DROP = re.compile(r"(--model-path|--host|--port|--log-level|--node-rank|--dist-init-addr) \S+")
LAYERS_235B, HEAD_DIM, BF16_BYTES = 94, 128, 2
RESOLVED = [
    ("tp", "tp_size"), ("ep", "ep_size"), ("attn-dp", "attn_dp_size"), ("nnodes", "nnodes"),
    ("quant", "quantization"), ("attention", "attention_backend"), ("moe runner", "moe_runner_backend"),
    ("moe a2a", "moe_a2a_backend"), ("chunked prefill", "chunked_prefill_size"), ("mem frac", "mem_fraction_static"),
    ("page", "page_size"), ("dispatch", "load_balance_method"),
]
EXTRA_FLAGS = [("ep_size", "--ep-size", 1), ("attn_dp_size", "--attn-dp-size", 1),
               ("moe_a2a_backend", "--moe-a2a-backend", "none")]


def passed_flags(arm_dir, args):
    cmd = Path(arm_dir) / "server_cmd.txt"
    if cmd.exists():
        text = cmd.read_text().strip()
        text = text[text.index("python3 -m sglang.launch_server"):] if "python3 -m sglang.launch_server" in text else text
        text = DROP.sub("", text.replace("python3 -m sglang.launch_server", "")).split()
        return " ".join(text), "recorded"
    flags = [f"--tp {args['tp_size']}", f"--kv-cache-dtype {args['kv_cache_dtype']}"]
    flags += [f"{flag} {args[key]}" for key, flag, default in EXTRA_FLAGS if args.get(key) != default]
    return " ".join(flags), "reconstructed"


def harness_commit(arm_dir):
    info = Path(arm_dir).parent / "job_info.txt"
    m = re.search(r"harness=([0-9a-f]{7})", info.read_text()) if info.exists() else None
    return m.group(1) if m else "?"


def kv_heads_per_gpu(lines, model):
    if "235B" not in model:
        return None
    for line in lines:
        m = KV_ALLOC.search(line)
        if m:
            k_bytes = float(m.group(3)) * 2**30 / int(m.group(2))
            return round(k_bytes / (LAYERS_235B * HEAD_DIM * BF16_BYTES), 2)
    return None


def collect(raw):
    groups = OrderedDict()
    for cfg, job, label, d in launches(raw):
        lines = read_lines(d / "server.log")
        args = server_args(lines)
        model = Path(args["model_path"]).name
        flags, how = passed_flags(d, args)
        resolved = tuple(str(args.get(k)) for _, k in RESOLVED)
        key = (cfg, model, flags, resolved)
        g = groups.setdefault(key, {"jobs": [], "how": set(), "kv_heads": set(), "harness": set()})
        g["harness"].add(harness_commit(d))
        g["jobs"].append(f"{job}{' ' + label if label else ''}")
        g["how"].add(how)
        g["kv_heads"].add(kv_heads_per_gpu(lines, model))
    return groups


def main(raw):
    groups = collect(raw)
    print("| Config | Checkpoint | Flags passed | Harness commit | Jobs (launch) |")
    print("|---|---|---|---|---|")
    for (cfg, model, flags, _), g in groups.items():
        how = "" if g["how"] == {"recorded"} else " (reconstructed)"
        harness = ", ".join(f"`{h}`" for h in sorted(g["harness"]))
        print(f"| {cfg} | `{model}` | `{flags}`{how} | {harness} | {', '.join(sorted(set(g['jobs'])))} |")
    print()
    print("| Config | " + " | ".join(h for h, _ in RESOLVED) + " | KV heads / GPU |")
    print("|---|" + "---|" * (len(RESOLVED) + 1))
    seen = set()
    for (cfg, _, _, resolved), g in groups.items():
        if (cfg, resolved) in seen:
            continue
        seen.add((cfg, resolved))
        kv = ", ".join(str(x) for x in sorted(v for v in g["kv_heads"] if v is not None)) or "—"
        print(f"| {cfg} | " + " | ".join(resolved) + f" | {kv} |")


if __name__ == "__main__":
    main(sys.argv[1])
