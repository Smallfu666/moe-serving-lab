# Post-hoc log analyses (CPU only, not preregistered, not scored)

Two questions asked after the scores, answered from the E1–E3 launch folders on CPU (no new GPU run). They change no
scored number or verdict in [`e1-results.md`](e1-results.md) to [`e4-results.md`](e4-results.md). The E4 Q3 numbers
had already been seen before these analyses were designed, so neither one is blind.

- Scripts: [`../analysis/prefill_decode_coupling.py`](../analysis/prefill_decode_coupling.py),
  [`../analysis/client_vs_engine.py`](../analysis/client_vs_engine.py) (shared parsing in
  [`../analysis/serverlog.py`](../analysis/serverlog.py)); deterministic, standard library only.
- Input: the raw launch folders (server logs, `phase_lines.txt`, replay records). They are private and not in this
  repository; the jobs are listed in [`../README.md`](../README.md#configurations-and-server-flags).
- Output, as committed: [`../analysis/derived/`](../analysis/derived/). The tables below are copied from it.
- Evidence kinds: **log** = SGLang's own log lines (1 s timestamps); **client** = the replay client's per-request
  timing (one `perf_counter` clock per launch); **inference** = our reading, not measured.

## A. Does decode slow down while prefill runs? (R16 against N4)

**Why.** R16 (16 GPUs, TP16 + attention-DP16 + EP16, round-robin dispatch) has TPOT p50 10.53 ms at c = 1 and
123.47 ms at c = 16 (client, table B), while its median per-rank log line holds one running request at every c
(E4 Q3). A larger batch does not explain that rise, so the question is whether decode on one rank slows while some
rank is prefilling.

**Measurement (log).** Each DP rank prints a `Decode batch` line every few decode steps, with `gen throughput` and
`#running-req`; per line, ms per token per request = 1000 × running ÷ gen throughput. A line covers the span from the
same rank's previous decode line to its own timestamp. Each line is classed by the `Prefill batch` lines (any rank,
#new-token > 0) in its span: **none** (no prefill on any rank), **other** (prefill only on other ranks), **own**
(prefill on this rank), **ambiguous** (prefill only in the boundary second). R16's four node logs are merged; N4 has
one scheduler, so it has no "other" class. Exact rules are in the script's docstring. N4-FP8 is the control: same
checkpoint, one node, no attention-DP.

Engine view (median ms per token per request; ranges over R16's 3 launches and N4's 4 replicas):

| Config | c | none: ms | none: lines | other: ms | other: lines | other: no-graph | other, graph lines only: ms | own: ms | own: lines | ambiguous: lines |
|---|---|---|---|---|---|---|---|---|---|---|
| R16-FP8 | 1 | 10.6 | 354–363 | 10.6–10.7 | 15–16 | 0.00 | 10.6–10.7 | 1550.5–1562.5 | 124 | 100–110 |
| R16-FP8 | 4 | 11.0 | 139–155 | 46.5–50.3 | 213–225 | 0.20–0.23 | 43.7–50.3 | 730.0–763.4 | 124–126 | 97–120 |
| R16-FP8 | 8 | 11.0–11.2 | 124–136 | 70.5–72.0 | 458–472 | 0.29–0.33 | 63.2–65.4 | 493.8–505.1 | 258–261 | 135–151 |
| R16-FP8 | 16 | 11.2 | 79–109 | 91.3–99.5 | 976–1019 | 0.38–0.41 | 71.8–79.8 | 290.7–301.2 | 545–547 | 189–203 |
| N4-FP8 | 1 | 7.5 | 303–322 | — | 0 | — | — | 7.6–7.8 | 87–93 | 217–231 |
| N4-FP8 | 4 | 8.5–8.6 | 66–74 | — | 0 | — | — | 10.7–10.9 | 54–58 | 82–87 |
| N4-FP8 | 8 | 7.8–8.6 | 58–63 | — | 0 | — | — | 12.9–13.2 | 72–75 | 77–82 |
| N4-FP8 | 16 | 9.7–9.8 | 19–24 | — | 0 | — | — | 16.0–16.2 | 113–117 | 78–82 |

A line's class depends on its own span length: a long span (a slow line) is more likely to contain a prefill, so
"none" holds only short spans and the "none" vs "other" medians above compare short spans with long ones. The
equal-span stratum compares R16 lines whose span is exactly 1 s, and gives the span length of all lines per class:

| Config | c | none: ms at 1 s | none: lines at 1 s | other: ms at 1 s | other: lines at 1 s | other ÷ none at 1 s | none: span median, max s | other: span median, max s |
|---|---|---|---|---|---|---|---|---|
| R16-FP8 | 1 | 10.6 | 132–134 | 10.6–10.7 | 15–16 | 1.0 | 0, 1 | 1, 1 |
| R16-FP8 | 4 | 11.0 | 49–55 | 22.0–22.4 | 78–94 | 2.0 | 0, 1 | 2, 7–10 |
| R16-FP8 | 8 | 10.9–11.1 | 35–43 | 21.5–23.1 | 104–106 | 2.0–2.1 | 0, 1 | 3, 9–10 |
| R16-FP8 | 16 | 11.1–11.2 | 11–26 | 20.7–30.0 | 141–165 | 1.8–2.7 | 0, 1 | 4, 10 |

**Cross-check (client).** For each request, the share of its decode interval [send + TTFT, send + e2e] during which at
least one other request of the same point was still waiting for its first token (that wait includes queueing as well
as prefill). Median TPOT ms (requests in brackets):

| Config | c | overlap 0 | overlap (0.0, 0.1] | overlap (0.1, 0.3] | overlap (0.3, 1.0] |
|---|---|---|---|---|---|
| R16-FP8 | 4 | 11.0 (27–28) | 10.7–11.0 (2) | 14.0 (0–2) | 41.8–43.9 (111–112) |
| R16-FP8 | 8 | 11.4–11.5 (19–23) | 11.0–11.4 (2–3) | 12.3–15.8 (0–1) | 80.8–85.3 (247–250) |
| R16-FP8 | 16 | 11.2 (19–21) | 11.1–11.7 (0–4) | 13.9–14.7 (1–2) | 127.9–134.5 (518–522) |
| N4-FP8 | 4 | 9.5–9.6 (31) | 10.0–10.1 (31) | 10.5–10.6 (70–71) | 12.8 (9–10) |
| N4-FP8 | 8 | 10.1–10.7 (17–19) | 11.1–11.3 (28–29) | 12.5–12.7 (124–126) | 14.5–14.7 (100–102) |
| N4-FP8 | 16 | 9.8 (21) | 11.3–11.5 (29–30) | 15.0–15.5 (155–160) | 18.8–19.1 (333–338) |

Where the overlap-0 requests sit (decode midpoint as a share of the point's span, and requests in flight then,
counting the request itself):

| Config | c | decode midpoint share | in flight at midpoint |
|---|---|---|---|
| R16-FP8 | 4 | 0.01–1.00 | 1–4 |
| R16-FP8 | 8 | 0.62–1.00 | 1–8 |
| R16-FP8 | 16 | 0.57–1.00 | 2–15 |
| N4-FP8 | 4 | 0.02–1.00 | 1–4 |
| N4-FP8 | 8 | 0.03–1.00 | 1–8 |
| N4-FP8 | 16 | 0.46–0.99 | 2–15 |


**What the data supports.**

- *Log.* On R16, lines whose span holds no prefill on any rank run at 10.6–11.2 ms per token at every c, about the
  c = 1 rate. At the same 1 s span, lines whose span holds prefill **only on other ranks** run at 20.7–30.0 ms
  (c = 4–16), 1.8–2.7× slower (equal-span table). Over all span lengths those lines run at 46.5–99.5 ms, but that
  4.2–8.9× is inflated by the selection on span length: "none" spans are at most 1 s long, "other" spans have a median
  of 2–4 s and reach 10 s. Graph lines alone in "other" spans are still 43.7–79.8 ms, so the slowdown is not only the
  no-graph lines. At c = 16 these spans hold more than half of R16's decode lines (976–1019), and
  clean spans 79–109.
- *Log.* No-graph decode lines occur only in spans that hold a prefill: in clean spans their share is 0 in 11 of the
  12 R16 launch-points (c = 1–16 × 3 launches; the exception is about one line in one c = 4 launch). So E4 Q3's rising no-graph fraction and this coupling are not two
  independent signals.
- *Log.* On N4, a prefill delays only its own scheduler: lines with an own prefill in the span run at 10.7–16.2 ms
  against 7.8–9.8 ms without (c = 4–16), up to 1.7×.
- *Client.* Requests whose decode overlapped other requests' waiting time for more than 30 % of it (most R16
  requests at c = 4, nearly all at c ≥ 8) have TPOT 41.8–134.5 ms, against 11.0–11.5 ms for the few with no
  overlap. On N4 the same split goes from 9.5–10.7 to 12.8–19.1 ms. At c = 4 the no-overlap requests are spread over the whole point; on R16 at
  c = 8 and 16 they all decode in the last 38–43 % of it, with as few as 1–2 requests in flight, so there the comparison
  is partly confounded with a lower effective concurrency.
- *Inference, not shown.* With attention-DP and EP, every rank takes part in each MoE layer's all-to-all, so a rank's
  decode step may have to wait while another rank runs a prefill pass. R16 also resolved `chunked_prefill_size` to
  1024 tokens (16384 on N4; logged `server_args`, see the configuration table), so one long prompt becomes many prefill
  passes, each of which could stall the decoding ranks. Neither part is traced.

**Not established.** How much of R16's 0.02028 / 0.02781 throughput ratio to N4 comes from this. There is no
counterfactual run, R16 also loses prefix reuse (E4 Q3), and each rank decodes about one request per step.

**Limits.** A line's class is selected on its own span length (above): slow lines have long spans and so are more
likely to hold a prefill. The equal-span stratum removes that for 1 s spans only, which hold 78–165 of R16's "other"
lines per launch-point at c ≥ 4; longer "other" spans are not compared with a "none" span of the same length, because
there is none. Timestamps have 1 s resolution and a decode line covers many steps, so the classes are coarse; a prefill
in the same second as a decode line can fall on either side of it. The four nodes' clocks agree to the second at the
first, synchronized prefill of every launch (all four logs show it in the same second), but finer skew is not
known. On R16 an "own" span can include time the rank sat idle before round-robin dispatch gave it a new request,
so the "own" ms (290–1563) is not a decode-step time and is not used above; an "other" span cannot include such idle
time, because a rank only gets a new request through its own prefill. Whether `gen throughput` counts the wall time
of prefill passes between two decode lines was not checked in the SGLang source. One SGLang nightly (`a6a8c560`),
one workload.

## B. Client-side vs engine-side timing

**Measurement.** *Client*: per-request TPOT = (e2e − TTFT) ÷ (output tokens − 1), timed by the replay client around
the streamed response; the p50 over requests per launch. *Engine (log)*: the median, over the point's `Decode batch`
lines, of 1000 × running ÷ gen throughput. Both are medians over launches (N4: replicas). They weigh differently
(requests vs log lines), so only c = 1, one request at a time, compares like with like.

| Config | c | launches | client TPOT p50 ms | engine median line ms | client − engine ms | client ÷ engine |
|---|---|---|---|---|---|---|
| E1 G-FP8 | 1 | 2 | 7.94 | 8.00 | -0.06 | 0.993 |
| E1 G-FP8 | 8 | 2 | 16.33 | 14.71 | 1.62 | 1.110 |
| E1 G-FP8 | 32 | 2 | 42.00 | 22.82 | 19.17 | 1.840 |
| E1 G-NV4 | 1 | 2 | 6.62 | 6.70 | -0.07 | 0.989 |
| E1 G-NV4 | 8 | 2 | 12.30 | 10.76 | 1.54 | 1.143 |
| E1 G-NV4 | 32 | 2 | 29.44 | 17.02 | 12.42 | 1.730 |
| E1 H-FP8 | 1 | 2 | 9.93 | 10.07 | -0.14 | 0.986 |
| E1 H-FP8 | 8 | 2 | 24.46 | 21.31 | 3.15 | 1.148 |
| E1 H-FP8 | 32 | 2 | 95.18 | 71.68 | 23.50 | 1.328 |
| E1 H-FP8-TP4 | 1 | 2 | 8.34 | 8.46 | -0.12 | 0.986 |
| E1 H-FP8-TP4 | 8 | 2 | 17.03 | 15.18 | 1.85 | 1.122 |
| E1 H-FP8-TP4 | 32 | 2 | 39.46 | 22.81 | 16.66 | 1.730 |
| G2-FP8 | 1 | 3 | 7.90 | 8.00 | -0.10 | 0.988 |
| G2-FP8 | 4 | 3 | 12.02 | 11.54 | 0.48 | 1.042 |
| G2-FP8 | 8 | 3 | 16.16 | 13.91 | 2.25 | 1.162 |
| G2-FP8 | 16 | 3 | 23.62 | 19.87 | 3.75 | 1.189 |
| G2-FP8 | 32 | 3 | 44.04 | 24.93 | 19.11 | 1.767 |
| G2-FP8 | 64 | 3 | 116.64 | 92.07 | 24.57 | 1.267 |
| H4-FP8 | 1 | 3 | 8.31 | 8.45 | -0.15 | 0.983 |
| H4-FP8 | 4 | 3 | 12.17 | 11.58 | 0.59 | 1.051 |
| H4-FP8 | 8 | 3 | 16.95 | 14.61 | 2.34 | 1.160 |
| H4-FP8 | 16 | 3 | 22.70 | 19.10 | 3.59 | 1.188 |
| H4-FP8 | 32 | 3 | 37.31 | 25.56 | 11.75 | 1.460 |
| H4-FP8 | 64 | 2 | 57.23 | 40.19 | 17.04 | 1.424 |
| G2-NV4 | 1 | 3 | 6.57 | 6.67 | -0.10 | 0.985 |
| G2-NV4 | 4 | 3 | 8.87 | 8.43 | 0.45 | 1.053 |
| G2-NV4 | 8 | 3 | 12.11 | 10.28 | 1.83 | 1.178 |
| G2-NV4 | 16 | 3 | 17.59 | 14.33 | 3.25 | 1.227 |
| G2-NV4 | 32 | 3 | 27.88 | 19.16 | 8.71 | 1.455 |
| G2-NV4 | 64 | 3 | 59.26 | 32.51 | 26.76 | 1.823 |
| H8-FP8 | 1 | 3 | 9.89 | 10.04 | -0.15 | 0.985 |
| H8-FP8 | 4 | 3 | 12.16 | 11.68 | 0.48 | 1.041 |
| H8-FP8 | 8 | 3 | 15.22 | 13.42 | 1.80 | 1.134 |
| H8-FP8 | 16 | 3 | 20.55 | 17.69 | 2.86 | 1.162 |
| H8-FP8 | 32 | 2 | 32.38 | 22.27 | 10.11 | 1.454 |
| H8-FP8 | 64 | 2 | 49.85 | 36.52 | 13.33 | 1.365 |
| H100-8-FP8 | 1 | 3 | 12.37 | 12.37 | 0.00 | 1.000 |
| H100-8-FP8 | 4 | 3 | 20.14 | 18.90 | 1.24 | 1.066 |
| H100-8-FP8 | 8 | 3 | 35.43 | 29.21 | 6.22 | 1.213 |
| H100-8-FP8 | 16 | 3 | 60.81 | 50.95 | 9.86 | 1.194 |
| H100-8-FP8 | 32 | 3 | 105.24 | 69.88 | 35.36 | 1.506 |
| N4-FP8 | 1 | 4 | 7.21 | 7.54 | -0.33 | 0.956 |
| N4-FP8 | 4 | 4 | 10.24 | 9.67 | 0.57 | 1.059 |
| N4-FP8 | 8 | 4 | 12.96 | 11.10 | 1.86 | 1.168 |
| N4-FP8 | 16 | 4 | 17.34 | 14.56 | 2.79 | 1.191 |
| N4-FP8 | 32 | 4 | 25.46 | 17.82 | 7.64 | 1.429 |
| N4-FP8 | 64 | 4 | 38.30 | 25.81 | 12.49 | 1.484 |
| N4-NV4 | 1 | 4 | 5.91 | 6.24 | -0.33 | 0.947 |
| N4-NV4 | 4 | 4 | 7.52 | 7.12 | 0.40 | 1.056 |
| N4-NV4 | 8 | 4 | 9.62 | 8.19 | 1.42 | 1.173 |
| N4-NV4 | 16 | 4 | 13.08 | 10.69 | 2.40 | 1.224 |
| N4-NV4 | 32 | 4 | 19.71 | 13.48 | 6.23 | 1.462 |
| N4-NV4 | 64 | 4 | 30.28 | 19.95 | 10.33 | 1.518 |
| R16-FP8 | 1 | 3 | 10.53 | 10.66 | -0.12 | 0.988 |
| R16-FP8 | 4 | 3 | 35.08 | 29.50 | 5.58 | 1.189 |
| R16-FP8 | 8 | 3 | 77.78 | 78.46 | -0.68 | 0.991 |
| R16-FP8 | 16 | 3 | 123.47 | 116.28 | 7.19 | 1.062 |

**What the data supports.**

- At c = 1 the two agree: client ÷ engine is 0.947–1.000 in every configuration and |client − engine| ≤ 0.33 ms.
  What the client adds on top of the engine (HTTP streaming,
  detokenization, the client's own event loop) is therefore not visible at c = 1 at this resolution.
- At c = 4–16 the client value is 1.04–1.23× the median log line outside R16; at c = 32–64 it is 1.27–1.84×. The ratio
  is larger at c ≥ 32 than at c = 4 in every configuration, though not always monotonic in c. It is not split here: the two weigh differently (a request's TPOT
  includes every slow stretch during its decode, while the median line is near the typical, faster stretch), and
  table A shows that stretches containing a prefill are slower. Client-side costs that grow with the number of
  concurrent streams cannot be excluded either.
- R16 is the exception (0.99–1.19 at c ≥ 4): most of its log lines are themselves in prefill-coupled spans (table A),
  so the median line already carries that slowdown.

**Which numbers the study's claims use.**

| Claim | Timing source |
|---|---|
| E1 TTFT p50, TPOT p50 (e.g. 7.943 / 9.932 / 6.625 ms at c = 1) | client (`summarize_launch.py`, field `tpot_s`) |
| E2 / E3 interactivity (tok/s/user) and throughput per GPU, and every ratio built from them | client (`e2_score.py`: per-request (n − 1) ÷ (e2e − TTFT); output tokens ÷ point wall time) |
| E4 per-token and per-step µs, class shares | nsys device time in profiled windows (`nsys_attrib.py`) |
| E4 G-DISTORT (a) | nsys step ÷ the same server's log step outside the window |
| E4 F-BRIDGE | nsys decode-only ratio against E2's client ratio |
| E4 Q3 per-line rates | log lines (as in table A) |

So every headline latency and throughput number in E1–E3 is client-side. At c = 1 it matches the engine's own rate;
at higher c it also contains time that is not decode steps, which is what users see and what the preregistrations
define, and which E4's F-BRIDGE already marks as "outside decode steps" for G2-NV4/H4 at 100 tok/s/user.

**Limits.** The log rate is SGLang's own estimate over a few decode steps, printed with two decimals; how its clock
treats prefill passes between lines was not checked in source. Points cut by the harness end time or aborted (no
`rc=0`) are left out: H4 l2 c = 64 (job 554270) and H8 l1 c = 32 (job 554979) ended with exit 20 and
`ServerDisconnectedError`, and H8 l1 c = 64 never ran, which is why those rows have 2 launches.
Launch-to-launch medians hide which launch drives a value; the per-launch numbers are recomputed by the script.
