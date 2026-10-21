"""Does decode slow down while prefill runs? R16-FP8 (16 GPUs, attention-DP16) against N4-FP8 (4 GPUs, no DP).

Usage: python prefill_decode_coupling.py <raw-root> [--md]   -> JSON on stdout (--md: summary tables, ranges over
launches or replicas).

Engine view (server logs, 1 s timestamps). Each DP rank prints a `Decode batch` line every few decode steps with
`gen throughput` and `#running-req`; per line, ms per token per request = 1000 × running ÷ gen throughput. A line
covers the span from the same rank's previous decode line (t_prev) to its own timestamp (t). It is classed by the
`Prefill batch` lines (any rank, #new-token > 0; R16 ranks are merged from the four node logs) in that span:
  none   no prefill line on any rank in [t_prev, t];
  other  prefill only on other ranks in (t_prev, t], none on this rank in [t_prev, t];
  own    a prefill line on this rank in (t_prev, t];
anything else (a prefill only in the boundary second t_prev) is "ambiguous". A rank's first line in a phase has no
span and is left out. N4 has one scheduler (DP rank 0), so it has no "other" class. On R16 an "own" span can include
time the rank sat idle before round-robin dispatch gave it a new request, so its ms is not a decode-step time; an
"other" span cannot, because a rank only gets a new request through its own prefill.

A line's class depends on its own span length: a long span (a slow line) is more likely to contain a prefill, so
"none" holds only short spans. The equal-span stratum compares the classes at one span length (EQUAL_SPAN_S).

Client view (replay records, one perf_counter clock per launch). For each request, the decode interval is
[t_send + TTFT, t_send + e2e]; its overlap is the fraction of that interval during which at least one other
request of the same point was waiting for its first token ([t_send, t_send + TTFT], which includes queueing as well
as prefill). Requests are binned by overlap and the median TPOT per bin is reported. For the overlap-0 requests the
position of their decode midpoint in the point and the number of requests in flight then are also reported, because
at high c such requests can only occur where fewer than c requests run (the tail of the point).
"""

import json
import statistics as st
import sys

from serverlog import events, launches, read_lines, replay_records, replay_windows, r16_peers

CONFIGS = ("R16-FP8", "N4-FP8")
POINTS = (1, 4, 8, 16)
BINS = ((0.0, 0.0), (0.0, 0.1), (0.1, 0.3), (0.3, 1.0))
EQUAL_SPAN_S = 1


def classify(span_prefills, rank, t_prev, t):
    inside = [p for p in span_prefills if t_prev < p["t"] <= t]
    at_prev = [p for p in span_prefills if p["t"] == t_prev]
    own = any(p["dp"] == rank for p in inside)
    other = any(p["dp"] != rank for p in inside)
    if own:
        return "own"
    if not inside and not at_prev:
        return "none"
    if other and not any(p["dp"] == rank for p in at_prev):
        return "other"
    return "ambiguous"


def engine_lines(log_lines_list, t0, t1):
    pf, dc = [], []
    for lines in log_lines_list:
        p, d = events(lines, t0, t1)
        pf += p
        dc += d
    pf = sorted((p for p in pf if p["new"] > 0), key=lambda p: (p["t"], p["dp"]))
    dc.sort(key=lambda d: (d["dp"], d["t"]))
    out, prev = [], {}
    for d in dc:
        rank = d["dp"]
        if rank in prev and d["running"] > 0 and d["gen_tput"] > 0:
            t_prev = prev[rank]
            span = [p for p in pf if t_prev <= p["t"] <= d["t"]]
            out.append({"cls": classify(span, rank, t_prev, d["t"]), "graph": d["graph"],
                        "ms": 1000.0 * d["running"] / d["gen_tput"],
                        "span_s": (d["t"] - t_prev).total_seconds()})
        prev[rank] = d["t"]
    return out


def summarize_engine(lines):
    res = {}
    for cls in ("none", "other", "own", "ambiguous"):
        sel = [x for x in lines if x["cls"] == cls]
        med = lambda xs: round(st.median(xs), 2) if xs else None
        res[cls] = {"lines": len(sel), "median_ms_per_token": med([x["ms"] for x in sel]),
                    "median_ms_graph_lines": med([x["ms"] for x in sel if x["graph"]]),
                    "median_ms_no_graph_lines": med([x["ms"] for x in sel if not x["graph"]]),
                    "no_graph_fraction": round(sum(not x["graph"] for x in sel) / len(sel), 3) if sel else None,
                    "median_span_s": med([x["span_s"] for x in sel]),
                    "max_span_s": max(x["span_s"] for x in sel) if sel else None,
                    f"lines_span_{EQUAL_SPAN_S}s": sum(x["span_s"] == EQUAL_SPAN_S for x in sel),
                    f"median_ms_span_{EQUAL_SPAN_S}s": med([x["ms"] for x in sel if x["span_s"] == EQUAL_SPAN_S])}
    return res


def overlap_fraction(rec, others):
    a, b = rec["t_send"] + rec["ttft_s"], rec["t_send"] + rec["e2e_s"]
    if b <= a:
        return None
    iv = sorted((max(a, o["t_send"]), min(b, o["t_send"] + o["ttft_s"])) for o in others)
    covered, end = 0.0, a
    for s, e in iv:
        if e <= s or e <= end:
            continue
        covered += e - max(s, end)
        end = e
    return covered / (b - a)


def summarize_client(recs):
    ok = [r for r in recs if not r.get("error") and r.get("ttft_s") is not None and r.get("tpot_s") is not None]
    t0 = min(r["t_send"] for r in ok)
    t1 = max(r["t_send"] + r["e2e_s"] for r in ok)
    rows, clean = [], []
    for i, r in enumerate(ok):
        f = overlap_fraction(r, ok[:i] + ok[i + 1:])
        if f is not None:
            rows.append((f, 1000.0 * r["tpot_s"]))
        if f == 0.0:
            mid = r["t_send"] + (r["ttft_s"] + r["e2e_s"]) / 2
            clean.append(((mid - t0) / (t1 - t0),
                          sum(o["t_send"] <= mid <= o["t_send"] + o["e2e_s"] for o in ok)))
    out = {}
    for lo, hi in BINS:
        sel = [ms for f, ms in rows if (f == 0.0 if hi == 0.0 else lo < f <= hi)]
        key = "0" if hi == 0.0 else f"({lo}, {hi}]"
        out[key] = {"requests": len(sel), "median_tpot_ms": round(st.median(sel), 2) if sel else None}
    if clean:
        out["0"].update({"decode_mid_share_min": round(min(x for x, _ in clean), 2),
                         "decode_mid_share_max": round(max(x for x, _ in clean), 2),
                         "in_flight_at_mid_min": min(n for _, n in clean),
                         "in_flight_at_mid_max": max(n for _, n in clean)})
    return out


def span(values, fmt="{:.1f}"):
    vals = [v for v in values if v is not None]
    if not vals:
        return "—"
    lo, hi = min(vals), max(vals)
    return fmt.format(lo) if fmt.format(lo) == fmt.format(hi) else f"{fmt.format(lo)}–{fmt.format(hi)}"


def markdown(result):
    print("Engine view: decode log lines by prefill in their span (median ms per token per request; lines; no-graph share)")
    print()
    print("| Config | c | none: ms | none: lines | other: ms | other: lines | other: no-graph | other, graph lines only: ms "
          "| own: ms | own: lines | ambiguous: lines |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for cfg in CONFIGS:
        runs = [v for k, v in result.items() if k.startswith(cfg + " ")]
        for c in POINTS:
            e = [r[f"c{c}"]["engine"] for r in runs if f"c{c}" in r]
            g = lambda cls, key: [x[cls][key] for x in e]
            print(f"| {cfg} | {c} | {span(g('none', 'median_ms_per_token'))} | {span(g('none', 'lines'), '{:d}')} "
                  f"| {span(g('other', 'median_ms_per_token'))} | {span(g('other', 'lines'), '{:d}')} "
                  f"| {span(g('other', 'no_graph_fraction'), '{:.2f}')} | {span(g('other', 'median_ms_graph_lines'))} "
                  f"| {span(g('own', 'median_ms_per_token'))} | {span(g('own', 'lines'), '{:d}')} "
                  f"| {span(g('ambiguous', 'lines'), '{:d}')} |")
    print()
    print(f"Equal-span stratum: R16 lines whose span is exactly {EQUAL_SPAN_S} s (median ms per token per request; lines), "
          "and the span length of all lines per class (median, max s)")
    print()
    print(f"| Config | c | none: ms at {EQUAL_SPAN_S} s | none: lines at {EQUAL_SPAN_S} s | other: ms at {EQUAL_SPAN_S} s "
          f"| other: lines at {EQUAL_SPAN_S} s | other ÷ none at {EQUAL_SPAN_S} s | none: span median, max s "
          "| other: span median, max s |")
    print("|---|---|---|---|---|---|---|---|---|")
    k_ms, k_n = f"median_ms_span_{EQUAL_SPAN_S}s", f"lines_span_{EQUAL_SPAN_S}s"
    runs = [v for k, v in result.items() if k.startswith("R16-FP8 ")]
    for c in POINTS:
        e = [r[f"c{c}"]["engine"] for r in runs if f"c{c}" in r]
        g = lambda cls, key: [x[cls][key] for x in e]
        ratio = [x["other"][k_ms] / x["none"][k_ms] for x in e if x["other"][k_ms] and x["none"][k_ms]]
        print(f"| R16-FP8 | {c} | {span(g('none', k_ms))} | {span(g('none', k_n), '{:d}')} | {span(g('other', k_ms))} "
              f"| {span(g('other', k_n), '{:d}')} | {span(ratio, '{:.1f}')} "
              f"| {span(g('none', 'median_span_s'), '{:g}')}, {span(g('none', 'max_span_s'), '{:g}')} "
              f"| {span(g('other', 'median_span_s'), '{:g}')}, {span(g('other', 'max_span_s'), '{:g}')} |")
    print()
    print("Client view: median TPOT ms (requests) by the share of a request's decode time during which another request "
          "was waiting for its first token")
    print()
    keys = ["0" if hi == 0.0 else f"({lo}, {hi}]" for lo, hi in BINS]
    print("| Config | c | " + " | ".join(f"overlap {k}" for k in keys) + " |")
    print("|---|---|" + "---|" * len(keys))
    for cfg in CONFIGS:
        runs = [v for k, v in result.items() if k.startswith(cfg + " ")]
        for c in POINTS[1:]:
            cl = [r[f"c{c}"]["client"] for r in runs if f"c{c}" in r]
            cells = [f"{span([x[k]['median_tpot_ms'] for x in cl])} ({span([x[k]['requests'] for x in cl], '{:d}')})"
                     for k in keys]
            print(f"| {cfg} | {c} | " + " | ".join(cells) + " |")
    print()
    print("Overlap-0 requests: where their decode midpoint falls in the point (share of the span from the first send to "
          "the last completion) and how many requests were in flight then, counting the request itself (min–max over "
          "requests and launches)")
    print()
    print("| Config | c | decode midpoint share | in flight at midpoint |")
    print("|---|---|---|---|")
    for cfg in CONFIGS:
        runs = [v for k, v in result.items() if k.startswith(cfg + " ")]
        for c in POINTS[1:]:
            z = [r[f"c{c}"]["client"]["0"] for r in runs if f"c{c}" in r]
            z = [x for x in z if x["requests"]]
            if not z:
                continue
            share = (f"{min(x['decode_mid_share_min'] for x in z):.2f}–{max(x['decode_mid_share_max'] for x in z):.2f}")
            fl = f"{min(x['in_flight_at_mid_min'] for x in z)}–{max(x['in_flight_at_mid_max'] for x in z)}"
            print(f"| {cfg} | {c} | {share} | {fl} |")


def main(raw, md=False):
    result = {}
    for cfg, job, label, d in launches(raw):
        if cfg not in CONFIGS:
            continue
        dirs = r16_peers(d) if cfg == "R16-FP8" else [d]
        logs = [read_lines(x / "server.log") for x in dirs]
        win = replay_windows(d, logs[0])
        for c in POINTS:
            if c not in win or win[c][2]:
                continue
            t0, t1, _ = win[c]
            key = f"{cfg} {job} {label}"
            result.setdefault(key, {})[f"c{c}"] = {
                "engine": summarize_engine(engine_lines(logs, t0, t1)),
                "client": summarize_client(replay_records(d, c)),
            }
    if md:
        markdown(result)
        return
    json.dump(result, sys.stdout, indent=1, sort_keys=True)
    print()


if __name__ == "__main__":
    main(sys.argv[1], "--md" in sys.argv[2:])
