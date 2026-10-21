"""Client-side TPOT against the server's own decode rate, per configuration and concurrency point.

Usage: python client_vs_engine.py <raw-root>   -> Markdown table on stdout.

Client: per-request TPOT = (e2e − TTFT) ÷ (output tokens − 1), timed by the replay client around the streamed HTTP
response (`replay_client.py`); its p50 is E1's TPOT, and its inverse per request is E2/E3 interactivity.
Engine: every `Decode batch` line in the point's server-log window gives ms per token per request
= 1000 × #running-req ÷ gen throughput; the engine value is the median over lines (R16: all 16 DP ranks).
The two are weighted differently (requests vs log lines), so only c = 1, where one request runs at a time, compares
like with like. Means over lines are not used: a line can also span time that is not decode steps (on R16, a rank's
idle time before its next request; see prefill_decode_coupling.py), which a median resists and a mean does not. Values are medians over a configuration's launches (N4: replicas)
of the per-launch value.
Points cut by the harness end time or aborted (no `rc=0` in arm_status.txt) are left out.
"""

import statistics as st
import sys
from collections import OrderedDict

from serverlog import events, launches, pct, read_lines, replay_records, replay_windows, r16_peers

SKIP = ("E1 REF", "E4 ")


def per_launch(d, cfg):
    dirs = r16_peers(d) if cfg == "R16-FP8" else [d]
    logs = [read_lines(x / "server.log") for x in dirs]
    out = {}
    for c, (t0, t1, cut) in replay_windows(d, logs[0]).items():
        recs = replay_records(d, c)
        if cut or not recs:
            continue
        ok = [r for r in recs if not r.get("error") and r.get("tpot_s") is not None]
        tpot = [1000.0 * r["tpot_s"] for r in ok]
        lines = [x for lg in logs for x in events(lg, t0, t1)[1] if x["running"] > 0 and x["gen_tput"] > 0]
        eng = [1000.0 * x["running"] / x["gen_tput"] for x in lines]
        out[c] = {"client_p50": pct(tpot, 0.5), "engine_median": st.median(eng) if eng else None}
    return out


def main(raw):
    table = OrderedDict()
    for cfg, job, label, d in launches(raw):
        if cfg.startswith(SKIP):
            continue
        for c, v in per_launch(d, cfg).items():
            table.setdefault((cfg, c), []).append(v)
    order = list(dict.fromkeys(cfg for cfg, _ in table))
    cols = ["client_p50", "engine_median"]
    print("| Config | c | launches | client TPOT p50 ms | engine median line ms | client − engine ms | client ÷ engine |")
    print("|---|---|---|---|---|---|---|")
    for (cfg, c), vals in sorted(table.items(), key=lambda kv: (order.index(kv[0][0]), kv[0][1])):
        m = {}
        for k in cols:
            xs = [v[k] for v in vals if v[k] is not None]
            m[k] = st.median(xs) if xs else None
        cl, en = m["client_p50"], m["engine_median"]
        tail = f"{en:.2f} | {cl - en:.2f} | {cl / en:.3f}" if en else "— | — | —"
        print(f"| {cfg} | {c} | {len(vals)} | {cl:.2f} | {tail} |")


if __name__ == "__main__":
    main(sys.argv[1])
