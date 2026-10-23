# Cache-aware routing: CPU study

Question: the main study summary proposes re-running the 16-GPU configuration (R16: one SGLang server, TP16 +
attention-DP16 + EP16, round-robin dispatch) with cache-aware dispatch, at about 8,000 SU (three 1.5-hour launches on 16 × GB200 plus a smoke test of about 800 SU; SU is the cluster's billing unit, 100 per
GB200 GPU-hour). Can the existing logs,
plus a validated simulation, answer that question without a GPU?

Short answer:
1. Inside one SGLang server, dispatch across DP ranks is not prefix-aware. The built-in policies are round-robin and
   load counts. The server does accept an explicit rank per request (`routed_dp_rank`), so prefix-aware dispatch
   inside one server has to come from outside: a router in DP-aware mode, or the client.
2. A trace-driven cache model reproduces the logged hit rates of E1–E3, including R16's 0.49–0.61 under
   round-robin across 16 DP ranks. Over 51 phase points the mean |error| is 0.27 pp and the max is 2.85 pp.
3. In simulation, session-sticky dispatch on R16 lifts the hit rate to 0.95–0.96 and cuts new prefill tokens to
   0.08–0.12 of round-robin.
4. An estimate from the logs puts R16 output throughput about 2.3–2.7× above the measured runs (c = 4–16). Per GPU,
   R16 would go from 0.04–0.06 of N4 to about 0.11–0.15. TPOT would still be 22–51 ms, above the 20 ms that
   50 tok/s/user needs. So by this estimate, lost prefix reuse is a real part of the R16 gap, but not most of it.
5. The cache question needs no GPU. A GPU run would only test the throughput estimate; the smallest useful one is in
   [GPU decision](#gpu-decision).

Labels: **[code]** read on SGLang `origin/main` `2e7c9d851c` (2026-10-21); **[log]** measured in the main study's launches;
**[sim]** output of the scripts here; **[estimate]** a model built on [log] and [sim].

## 1. What SGLang routes, and where [code]

Inside one server (`python/sglang/srt/managers/data_parallel_controller.py`):
- **Policies.** `LoadBalanceMethod` has ROUND_ROBIN, FOLLOW_BOOTSTRAP_ROOM, TOTAL_REQUESTS and TOTAL_TOKENS
  (`:104-110`).
  - `--load-balance-method` accepts `auto, round_robin, follow_bootstrap_room, total_requests, total_tokens`
    (`arg_groups/fields/parallel.py:91-103`).
  - `auto` resolves to round_robin except on a PD prefill server (`arg_groups/serving_hook.py:215-228`). The R16
    log records `load_balance_method: 'round_robin'`.
  - TOTAL_REQUESTS and TOTAL_TOKENS pick the rank with the fewest running + waiting requests, or the fewest
    tokens, read from shared-memory load snapshots. They then add the new request to that rank's count
    (`:121-154`, `:886-899`).
  - The token count includes KV-pool usage, but no policy matches the prompt against a rank's radix cache. **No
    in-server policy is prefix-aware.**
- **External choice of rank.** Every scheduler first checks `req.routed_dp_rank` and, if it is set, sends the
  request straight to that rank (`maybe_external_dp_rank_routing`, `:839-852`; for example `:854-856`).
  - The field is on `GenerateReqInput` (`managers/io_struct.py:297-302`; also `:298` in the pinned `a6a8c56015b8`).
  - It is range-checked in `tokenizer_manager.py:930-939`.
  - So a client or a router can pin a session to a rank.

Across replicas, the repo has two routers:
- **`sgl-model-gateway`** (the `sglang_router` package).
  - **Defaults:** policy `cache_aware`, `cache_threshold` 0.3, `balance_abs_threshold` 64, `balance_rel_threshold`
    1.5, eviction every 120 s, `max_tree_size` 2^26 (`src/main.rs:150-171`). The Python bindings use the same
    values, except that eviction is every 60 s (`bindings/python/src/sglang_router/router_args.py:59-63`).
  - **Load fallback.** In `select_worker` (`src/policies/cache_aware.rs:387-490`), load is the router's own
    in-flight count per worker. When `max_load - min_load > abs` **and** `max_load > rel × min_load`, it picks the
    least-loaded worker (`:404-424`).
  - **Prefix match.** Otherwise it matches the request text against one approximate radix tree. The tree records
    which worker each routed prompt went to: it is built from routing history, not from the engines' caches
    (`:19-32`).
    - If matched chars / prompt chars > threshold, the request goes to that node's tenant; otherwise to the
      least-loaded worker. Either way, the prompt is then inserted under the chosen worker (`:426-471`).
    - When several tenants share the deepest node, the tree returns the node's cached `last_tenant`, which was set
      when the node was created (`src/policies/tree.rs:568-599`). Ties are therefore sticky, not balanced.
  - **DP-aware mode.** `--dp-aware` expands each server URL into one worker per DP rank
    (`core/steps/worker/local/create_worker.rs:297-335`). Each such worker writes `data_parallel_rank` into the
    request (`src/core/worker.rs:1106-1111`). This is what makes the gateway prefix-aware across the DP ranks of a
    single server.
  - **Unverified:** what text it extracts from a native `/generate` request that carries only `input_ids`, which
    is what the study's replay client sends. The protocol crate is outside this repo.
- **`sgl-router`** ("Slim, KV-aware router", `README.md:3`).
  - Its `cache_aware` prefers the engine holding the longest **actually cached** prefix, known from engine KV
    events through a local or external indexer, with at least 1024 matched tokens by default. `session_aware` pins
    by a session header. Both fall back to `power_of_two`, the documented default "under reorg"
    (`README.md:127-149`).
  - `--dp-aware` also picks the DP rank inside an engine (`README.md:331-335`): a sticky key hashes to a fixed
    rank; otherwise the rank with the deepest cached prefix; otherwise the fewest in-flight requests
    (`src/policies/dp_rank.rs:4-53`).
  - Read, not simulated.

So routing inside one server can be prefix-aware, but only with help from outside: a router in DP-aware mode, or a
client that sets `routed_dp_rank`. The engine never chooses by prefix on its own. This is the same split as in
[DeepInfer](https://github.com/deepseek-ai/DeepInfer), whose Entrypoint matches prefixes before dispatching to DP replicas (from its architecture guide, not
checked in its code here).

## 2. Method

### Inputs

- **Replay file.** Rebuilt on CPU with `main-study/harness/load_trajectories.py` from the public dataset parquet and
  the pinned tokenizer. Its sha256 `baf4efba…d1f7ff` equals `input_sha.txt` of every E1–E3 launch, so the token ids
  are byte-identical to what the servers received. `workload.py` refuses any other file.
- **Launch logs.** `replay_c*.jsonl`, `phase_lines.txt` and `server.log`, kept in a raw folder outside this repo.
  Scripts and results name launches by label (`E3 R16-FP8 L1`); a private map (`$LAUNCH_MAP`) gives each label its
  directory.
- **KV capacity** is `max_total_num_tokens` from `server.log`:

  | Configuration | Tokens |
  |---|---|
  | 2 × GB200 TP2 FP8 | 514,048 |
  | 2 × GB200 TP2 NVFP4 | 1,019,072 |
  | 2 × H200 TP2 | 160,140 |
  | 4 × H200 TP4 | 1,522,852 |
  | 4 × GB200 TP4 | 2,235,840 |
  | R16, per DP rank | 539,136 |

- **Page size** is 64 on GB200, which `trtllm_mha` forces (every `cached_tokens` in the R16 L2 c = 16 phase is a
  multiple of 64), and 1 on H200.

### Cache model

`prefix_cache.py` is a page-granular radix cache like SGLang's `RadixCache`:
- A sequence is a chain of page keys, and only whole pages are kept.
- A running request locks its pages; least-recently-used unlocked leaves are evicted.
- A finished request leaves its prompt pages and its output pages. The output pages never match a later prompt,
  because the replay sends the recorded answer, not the server's own output.

### Validation

`validate.py` feeds every logged replay request in `t_send` order:
- A request takes its pages at `t_send + ttft` and releases them at `t_send + e2e`.
- Caches start empty in every phase, because the harness calls `/flush_cache` before each phase (`run_arm.sh`,
  `flush_wait`).
- R16 is modelled as 16 caches fed round-robin.
- Only logged timing is used, never the logged `cached_tokens`.
- The rule "pages at first token" was chosen after comparing it with "pages at send time". It matches SGLang, which
  queues a request before its prefill. Both variants are reported.

### Routing simulation

`routing_sim.py` is a closed loop like `replay_client.py`:
- C workers run 2C sessions in total, starting at session 52; E2/E3 phases ran C sessions.
- Each replica has a FIFO prefill queue at R new tokens/s, a TTFT intercept, and decode of D × TPOT.
- R and the intercept are fitted on c = 1 requests (`calibrate.py`): G2-FP8 27,601 tok/s (R² 0.86), R16 4,362 tok/s
  per request (R² 0.99), N4 45,081 tok/s. TPOT is the logged p50 at the per-replica concurrency.
- Policies:
  - `round_robin`.
  - `least_load`: fewest in-flight requests.
  - `session`: sha256 of the session id, mod N.
  - `session_lb`: a session is bound at its first turn to the least-loaded replica, then kept there.
  - `cache_aware`: the gateway logic above with its defaults; `cache_aware_abs4` is the same with
    `balance_abs_threshold` 4.
  - `oracle`: least prefill work ahead, given the true cache contents. It is greedy, not optimal.
- Each policy runs with three seeds; `least_load`, `session_lb`, `cache_aware` and `oracle` break ties at random.
  The seed spread (`hit_min`/`hit_max` in `results/routing.json`) reaches about 3 pp under capacity pressure, so
  differences of that size between policies are noise.

### Log-only scripts

- `coupling.py` estimates how much of each request's decode window overlapped other requests' prefill. A prefill
  window is taken as new tokens / R before the first token, so the windows are [estimate], built on [log] timestamps.
- `project_r16.py` turns that overlap into a throughput estimate for sticky R16.

### Running it

`REPLAY=<replay.jsonl> RAW=<raw folder> LAUNCH_MAP=<label map> bash run_all.sh` runs on CPU in about 5 minutes on 16 cores. All tables are
in [`results/tables.md`](results/tables.md).

## 3. Validation [sim vs log]

| Configuration | Phases | Hit rate measured → simulated, error range |
|---|---|---|
| E1 G-FP8, 2×GB200 (L1, L2) | c1, c8, c32 | 0.958→0.957 … 0.926→0.918; −0.86 to +0.12 pp |
| E1 H-FP8, 2×H200 (L1, L2) | c1, c8, c32 | capacity-limited c32: 0.308→0.308, 0.313→0.313; −0.05 to +0.10 pp |
| E2 G2-FP8 / G2-NV4 / H4, E3 N4 replica | c1…c64 | −2.85 (G2-NV4 c64) to +0.61 pp |
| E3 R16, 16 DP ranks round-robin (L1–L3) | c1…c32 | 0.487→0.487, 0.557–0.608 → 0.562–0.605; −0.34 to +1.06 pp |

Over all 51 phase points, mean |error| is 0.27 pp and the max is 2.85 pp. The points are not independent:
- Launches of one arm repeat the same requests, and many c ≤ 16 phases sit near the 0.96 ceiling.
- The informative points are the nine non-R16 phases with measured hit < 0.93 (KV pool short of the working set),
  with errors −2.85, −1.21, −0.86, −0.86, +0.10, +0.09, +0.05, −0.04 and +0.01 pp, and the 15 R16 points, with errors from −0.34 to +1.06 pp.
- With pages taken at send time instead of first token, the error on the capacity-limited H200 c32 phase reaches
  65 pp.

So round-robin over 16 separate caches fully explains R16's hit rate. Every session's prompt starts with the same
5,715-token system-plus-tools prefix (counted from the replay input, which is not in this repo), and every rank soon holds it. A turn mostly lands on a rank that has seen
neither that session's previous turn nor its recent ones.

New prefill tokens:
- **[sim]** Phase-matched (c = 4, 8, 16), round-robin computes 9.3, 8.8 and 8.5× the new prefill tokens of `session_lb`.
- **[log]** R16 computed 9.5, 9.6 and 9.7× the new prefill tokens of one N4 replica, which hits at ≈ 0.96 (per-point
  counts from the raw logs, not in this repo; the E4 Q3 range over c = 1–16 is 9.4–13×, in
  [`e4-results.md`](../../main-study/results/e4-results.md)).

The service-time model is rougher ([`results/timing_check.json`](results/timing_check.json)):

| Quantity | Simulated vs logged |
|---|---|
| Hit rate | within 3.8 pp |
| TTFT p50 | 10–41 % low, except at the capacity-limited G2-FP8 c64 and R16 c32 points (25 % and 11 % high) |
| Output tok/s, G2-FP8 | 5–30 % low |
| Output tok/s, R16 | 8–52 % low |

The likely reasons for the tok/s gap are the constant TPOT p50 and the lack of prefill–decode coupling in this
model. Neither was tested. The routing grid below is therefore quoted for hit rates and prefill tokens only.

## 4. Results [sim]

Hit rate by policy, with prefill tokens computed relative to round-robin in brackets. Values are the mean of 3
seeds. The upper bound, the hit rate of one unbounded cache, is 0.955–0.960.

| Replicas | C | round_robin | least_load | session | session_lb | cache_aware (gateway defaults) | cache_aware abs=4 | oracle |
|---|---|---|---|---|---|---|---|---|
| 2 × G2-FP8 | 16 | 0.915 | 0.941 (0.69×) | 0.955 (0.53×) | 0.955 (0.53×) | 0.955 (0.52×) | 0.955 (0.53×) | 0.954 (0.55×) |
| 2 × G2-FP8 | 32 | 0.880 | 0.943 (0.48×) | 0.957 (0.36×) | 0.957 (0.36×) | 0.751 (2.08×) | 0.957 (0.36×) | 0.955 (0.38×) |
| 2 × G2-FP8 | 64 | 0.559 | 0.713 (0.65×) | 0.676 (0.73×) | 0.718 (0.64×) | 0.345 (1.49×) | 0.780 (0.50×) | 0.703 (0.67×) |
| 4 × G2-FP8 | 32 | 0.805 | 0.925 (0.38×) | 0.957 (0.22×) | 0.957 (0.22×) | 0.785 (1.10×) | 0.957 (0.22×) | 0.955 (0.23×) |
| 4 × G2-FP8 | 64 | 0.646 | 0.921 (0.22×) | 0.959 (0.12×) | 0.959 (0.12×) | 0.349 (1.84×) | 0.953 (0.13×) | 0.955 (0.13×) |
| 4 × G2-FP8 | 128 | 0.429 | 0.664 (0.59×) | 0.685 (0.55×) | 0.689 (0.55×) | 0.759 (0.42×) | 0.730 (0.47×) | 0.662 (0.59×) |
| 8 × G2-FP8 | 64 | 0.605 | 0.903 (0.25×) | 0.959 (0.10×) | 0.959 (0.10×) | 0.354 (1.63×) | 0.862 (0.35×) | 0.957 (0.11×) |
| 8 × G2-FP8 | 128 | 0.477 | 0.899 (0.19×) | 0.959 (0.08×) | 0.959 (0.08×) | 0.795 (0.39×) | 0.950 (0.10×) | 0.955 (0.09×) |
| 8 × G2-FP8 | 256 | 0.380 | 0.661 (0.55×) | 0.750 (0.40×) | 0.715 (0.46×) | 0.836 (0.26×) | 0.754 (0.40×) | 0.681 (0.51×) |
| R16, 16 DP ranks | 16 | 0.590 | 0.861 (0.34×) | 0.952 (0.12×) | 0.951 (0.12×) | 0.955 (0.11×) | 0.778 (0.54×) | 0.951 (0.12×) |
| R16, 16 DP ranks | 32 | 0.600 | 0.860 (0.35×) | 0.955 (0.11×) | 0.955 (0.11×) | 0.843 (0.39×) | 0.774 (0.56×) | 0.955 (0.11×) |
| R16, 16 DP ranks | 64 | 0.559 | 0.874 (0.29×) | 0.958 (0.10×) | 0.958 (0.10×) | 0.351 (1.47×) | 0.761 (0.54×) | 0.958 (0.10×) |
| R16, 16 DP ranks | 128 | 0.458 | 0.878 (0.22×) | 0.959 (0.08×) | 0.959 (0.08×) | 0.796 (0.38×) | 0.734 (0.49×) | 0.958 (0.08×) |

Load imbalance is in [`results/tables.md`](results/tables.md), measured as max/mean over replicas, both of prefill
tokens computed and of tokens held by in-flight requests. The pattern:

**Round-robin** is balanced (prefill imbalance 1.01–1.34) and wastes the most prefill: its hit rate is 0.38–0.92.

**Sticky by session** reaches the upper bound whenever each replica can hold its share of sessions.
- With a plain hash, few sessions spread unevenly. On R16 at C = 16 the imbalance is 1.83 for prefill and 3.83 for
  held tokens. `session_lb` binds each new session to the least-loaded rank and brings this to 1.50 and 1.76, at
  the same hit rate (grid shape: 2C = 32 sessions over 16 ranks).
- Under capacity pressure (32 sessions in flight per 514k-token G2 replica: 2×64, 4×128, 8×256), every policy
  except collapsed `cache_aware` lands at 0.66–0.84, against 0.38–0.56 for round-robin.

**Gateway `cache_aware` with its defaults collapses onto one replica** (imbalance = N) whenever in-flight requests
stay under `balance_abs_threshold` 64.
- The shared 5,715-token prefix is 0.83 of a median first-turn prompt (also from the replay input), above `cache_threshold` 0.3. So every new
  session follows the first tenant of that prefix.
- When that replica runs out of KV, the hit rate falls to 0.35.
- Only at C = 128–256, where the absolute threshold trips, does it spread load. There it beats the other policies
  under capacity pressure (0.76–0.84).
- With `balance_abs_threshold` 4 it matches session-sticky on 2 and 4 G2 replicas below capacity. It does worse at
  8 × 64 (0.862) and on R16 (0.73–0.78), where 4 in-flight requests is a large share of C/16 and rebalancing breaks
  affinity.

**least_load** keeps part of the affinity by accident, because the worker's previous request has just finished: its
hit rate is 0.66–0.94.

The **oracle** (greedy, not optimal) is no better than session-sticky below capacity. There, a session
id is all a router needs. Under capacity pressure the results are mixed: the oracle beats `session` at 2 × 64
(0.703 vs 0.676) but not at 4 × 128 or 8 × 256, and only `cache_aware` (abs = 4, or gateway defaults at C ≥ 128)
is clearly above it.

## 5. What it would mean for R16 [estimate]

### Decode under prefill

The overlap windows below are [estimate]; the TPOTs are [log] ([`results/coupling.json`](results/coupling.json)).
Totals are over 3 R16 launches.

| c | TPOT p50, overlap < 0.1 | TPOT p50, overlap > 0.9 |
|---|---|---|
| 4 | 11.0 ms (n = 84) | 87 ms (n = 38) |
| 8 | 11.5 ms (n = 72) | 125 ms (n = 193) |
| 16 | 11.2 ms (n = 63) | 167 ms (n = 611) |

Overlap is the share of a request's decode window during which some other request was prefilling.
- At c = 1, R16 TPOT is 10.5 ms.
- On N4, overlap averages 3–9 %. Low-overlap TPOT rises from 10.2 to 15.7 ms between c = 4 and 16.

So on R16 the TPOT rise goes with prefill running somewhere in the server much more than with batch size. This is a
correlation, not an ablation. The low-overlap groups are small: about 4 % of requests at c = 16.

### Throughput with sticky dispatch

`project_r16.py` combines the overlap relation with the [sim] new-token ratio of `session_lb` against round-robin
at the same phase shape (0.107–0.117).
- **Model.** Wall time = per-worker output × TPOT(overlap) + per-worker prefill time. The overlap share is prefill
  seconds times a factor fitted on each measured round-robin phase.
- **Fit check.** On the round-robin phases themselves, the model's wall time is 2–18 % high per launch.
- **Not modelled.** Rank collisions of the sticky assignment.

| c | measured R16 out tok/s | projected, sticky | vs measured (model vs model) | projected TPOT | R16/N4 per GPU, measured → projected |
|---|---|---|---|---|---|
| 4 | 76.3 | 175 | 2.3× (2.6×) | 22 ms | 0.063 → 0.145 |
| 8 | 85.7 | 217 | 2.5× (2.7×) | 36 ms | 0.047 → 0.120 |
| 16 | 114.2 | 307 | 2.7× (2.8×) | 51 ms | 0.040 → 0.107 |

Reading: by this estimate, fixing dispatch wins about 2.3–2.7× on R16. It leaves R16 near 0.1 of N4 per GPU, with
TPOT above 20 ms, so the E3 ratio at 50 tok/s/user (0.020) would stay small.

A hypothesis for the rest, not tested here: the 4-node EP16 configuration itself. A single request prefills at
4,362 tok/s on R16, against 27,601 tok/s on two GB200s. And prefill on one rank appears to slow decode on all ranks.

## GPU decision

- **The cache question needs no GPU.** It asks how much prefix reuse R16 loses and what prefix-aware dispatch
  recovers, and the validated simulation answers both. On the [estimate] above, the planned R16 re-run (about 8,000 SU) is
  not justified.
- **A GPU run would only test the throughput estimate in §5.** The smallest useful run is one R16 launch in the
  pinned image:
  - The replay client sets `routed_dp_rank`, binding each new session to the next rank in first-seen order mod 16.
    This is a balanced assignment; a plain hash would put c = 16 sessions on only 9 ranks. The simulation's
    `session_lb` binds to the least-loaded rank instead, which is equally balanced at the start of a phase. No router is needed: the
    field exists in `a6a8c56015b8`.
  - Phases c = 4, 8 and 16 only, with no GSM8K or logprob phases.
  - Compared against the three existing round-robin launches.
  - About 1 h on 16 × GB200 ≈ **1,600 SU** at 100 SU/GPU-h, plus a separately capped smoke test.
  - Prediction to test ([sim]): hit rate ≥ 0.94 at every point. A measured speed-up outside 1.5–4× refutes §5.
- **Cheaper alternative.** Add the same sticky phases to a planned 8 × H200 attention-DP8 + EP8 launch, for about
  +0.5 h × 8 GPUs × 60 SU/GPU-h ≈ **240 SU**. This tests the mechanism (prefill coupling under DP attention), not the
  4-node number.

## Limits

- **One workload, with recorded answers.** The server's own output never matches later turns, so real serving
  reuses more (the serving bound). All sessions also share a long system prefix; other agents and tool sets will
  differ.
- **Cache model.** No partial caching of chunked prefill, no retraction, no hierarchical cache.
- **Routing-grid timing.** It is a FIFO-prefill, constant-TPOT model that does not couple prefill and decode, so it
  ranks policies by hit rate and prefill tokens only.
- **`cache_aware`.** It re-implements the gateway's logic over token pages instead of characters, with
  `first_owner` standing in for `last_tenant`. `sgl-router` (actual KV events, admission caps, `power_of_two`) was
  read but not simulated.
- **§5 is an estimate, not a measurement.** It rests on a correlation (overlap against TPOT) and a one-parameter fit.
