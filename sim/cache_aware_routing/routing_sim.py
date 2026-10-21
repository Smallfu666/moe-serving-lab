"""Closed-loop routing simulation of the agentic replay over N replicas, each with its own prefix cache.

C workers each take the next session and send its turns one after another, as `replay_client.py` does. A router
picks a replica per request. Each replica has a FIFO prefill queue served at R new tokens/s; a request takes its cache
pages when its prefill starts (and waits while eviction cannot make room), and its decode lasts D x TPOT, after which
it releases the pages and the worker sends the next turn. R, the TTFT intercept and TPOT come from
`results/calibration.json`; capacity and page size from the launch logs. Decode does not slow down while a prefill
runs, a simplification that favours policies which compute more prefill.

Load imbalance is reported two ways: total prefill tokens computed per replica (max / mean over replicas), and the
prompt + decode tokens held by in-flight requests per replica (max / mean, averaged over dispatch instants).

Policies:
  round_robin   SGLang's in-server default (`data_parallel_controller.py` round_robin_scheduler)
  least_load    fewest in-flight requests, random tie-break (SGLang total_requests; gateway shortest queue)
  session       sha256(session id) mod N (sgl-router dp_rank.rs affinity key)
  session_lb    a session is bound at its first turn to the replica with the fewest in-flight requests, then kept
                there (a balanced sticky assignment, as a client or router binding table would do)
  cache_aware   sgl-model-gateway cache_aware.rs: if max-min in-flight > abs and max > rel x min, least load; else
                the longest prefix in the router-side tree; if match/prompt > threshold, that node's tenant, else
                least load; the chosen replica then joins the tree for the whole prompt. The tree holds routed
                prompts only and is not trimmed (default max_tree_size 2^26 is far above this workload), so it can
                believe a prefix is cached after the replica evicted it. When several replicas reach the deepest
                node, the gateway returns that node's cached `last_tenant` (tree.rs), which is sticky; here, the
                replica that reached the node first.
  oracle        perfect knowledge of every cache at dispatch: the replica with the least prefill work ahead
                (uncached tokens queued there plus this prompt's uncached tokens on it), ties to least load. Greedy,
                so not an optimum.
"""

import argparse
import hashlib
import heapq
import itertools
import json
import multiprocessing as mp
import random
import statistics
from pathlib import Path

from prefix_cache import PrefixCache
from workload import Sessions, request_keys

MAX_ANSWER = 1024
_G = {}


def _one(args):
    idx, c, n, cap, page, rate, icpt, tpot, pol, seed, kw = args
    return simulate(_G["sessions"], idx, c, n, cap, page, rate, icpt, tpot, pol, seed, **kw)


class Replica:
    def __init__(self, capacity, page):
        self.cache = PrefixCache(capacity, page)
        self.queue = []
        self.busy = False
        self.blocked = False
        self.inflight = 0
        self.prefill_tokens = 0
        self.pending = 0
        self.held_tokens = 0
        self.tree = set()


class Router:
    def __init__(self, policy, n, rng, cache_threshold=0.3, abs_thr=64, rel_thr=1.5):
        self.policy, self.n, self.rng = policy, n, rng
        self.thr, self.abs_thr, self.rel_thr = cache_threshold, abs_thr, rel_thr
        self.rr = 0
        self.first_owner = {}
        self.binding = {}

    def least_load(self, reps, among=None):
        idx = range(self.n) if among is None else among
        m = min(reps[i].inflight for i in idx)
        return self.rng.choice([i for i in idx if reps[i].inflight == m])

    @staticmethod
    def prefix_in(keyset, keys):
        n = 0
        for k in keys:
            if k not in keyset:
                break
            n += 1
        return n

    def route(self, req, reps):
        p = self.policy
        if p == "round_robin":
            self.rr += 1
            return (self.rr - 1) % self.n
        if p == "least_load":
            return self.least_load(reps)
        if p == "session":
            return int(hashlib.sha256(req["session"].encode()).hexdigest()[:16], 16) % self.n
        if p == "session_lb":
            if req["session"] not in self.binding:
                self.binding[req["session"]] = self.least_load(reps)
            return self.binding[req["session"]]
        if p == "oracle":
            cost = [r.pending + req["prompt"] - min(r.cache.match(req["prompt_keys"]) * req["page"], req["prompt"])
                    for r in reps]
            best = min(cost)
            return self.least_load(reps, [i for i in range(self.n) if cost[i] == best])
        if p.startswith("cache_aware"):
            loads = [r.inflight for r in reps]
            if max(loads) - min(loads) > self.abs_thr and max(loads) > self.rel_thr * min(loads):
                i = self.least_load(reps)
            else:
                best = max(self.prefix_in(r.tree, req["prompt_keys"]) for r in reps)
                if best and best * req["page"] / req["prompt"] > self.thr:
                    i = self.first_owner[req["prompt_keys"][best - 1]]
                else:
                    i = self.least_load(reps)
            reps[i].tree.update(req["prompt_keys"])
            for k in req["prompt_keys"]:
                self.first_owner.setdefault(k, i)
            return i
        raise ValueError(p)


def simulate(sessions, sess_idx, conc, n_rep, capacity, page, rate, icpt, tpot, policy, seed, **kw):
    rng = random.Random(seed)
    reps = [Replica(capacity, page) for _ in range(n_rep)]
    router = Router(policy, n_rep, rng, **kw)
    ev, tick = [], itertools.count()
    todo = list(sess_idx)
    out, samples = [], []
    uid = itertools.count()

    def push(t, kind, data):
        heapq.heappush(ev, (t, next(tick), kind, data))

    def next_turn(w, t, si, k):
        while si is None or k >= len(sessions.turns(si)):
            if not todo:
                return
            si, k = todo.pop(0), 0
        p, ans = sessions.turns(si)[k]
        d = min(ans, MAX_ANSWER)
        sk = sessions.keys(si, page)
        u = next(uid)
        req = {"w": w, "si": si, "k": k, "session": sessions.session_id(si), "prompt": p, "decode": d,
               "page": page, "t_send": t, "prompt_keys": sk[:p // page],
               "keys": request_keys(sk, p, p + d, page, u)}
        r = router.route(req, reps)
        req["rep"] = r
        req["est_new"] = p - min(reps[r].cache.match(req["prompt_keys"]) * page, p)
        reps[r].pending += req["est_new"]
        reps[r].inflight += 1
        reps[r].held_tokens += p + d
        reps[r].queue.append(req)
        held = [x.held_tokens for x in reps]
        samples.append(max(held) / statistics.fmean(held))
        try_start(r, t)

    def try_start(r, t):
        rep = reps[r]
        if rep.busy or not rep.queue:
            return
        req = rep.queue[0]
        hit = rep.cache.admit(req["keys"], t, strict=True)
        if hit is None:
            rep.blocked = True
            return
        rep.blocked = False
        rep.queue.pop(0)
        req["cached"] = min(hit * page, req["prompt"])
        new = req["prompt"] - req["cached"]
        rep.prefill_tokens += new
        rep.busy = True
        req["t_start"] = t
        push(t + new / rate, "prefill_done", req)

    for w in range(conc):
        next_turn(w, 0.0, None, 0)
    t = 0.0
    while ev:
        t, _, kind, req = heapq.heappop(ev)
        rep = reps[req["rep"]]
        if kind == "prefill_done":
            req["ttft"] = t - req["t_send"] + icpt
            rep.pending -= req["est_new"]
            rep.busy = False
            push(t + req["decode"] * tpot, "finish", req)
            try_start(req["rep"], t)
        else:
            rep.cache.release(req["keys"], t)
            rep.inflight -= 1
            rep.held_tokens -= req["prompt"] + req["decode"]
            out.append(req)
            if rep.blocked:
                try_start(req["rep"], t)
            next_turn(req["w"], t, req["si"], req["k"] + 1)
    prompt = sum(r["prompt"] for r in out)
    cached = sum(r["cached"] for r in out)
    pf = [x.prefill_tokens for x in reps]
    ttft = sorted(r["ttft"] for r in out)
    return {"requests": len(out), "hit": cached / prompt, "prefill_tokens": prompt - cached,
            "prefill_imbalance": max(pf) / statistics.fmean(pf),
            "inflight_tokens_max_over_mean": statistics.fmean(samples),
            "ttft_p50_s": ttft[len(ttft) // 2], "ttft_p90_s": ttft[int(len(ttft) * 0.9)],
            "output_tok_per_s": sum(r["decode"] for r in out) / t, "makespan_s": t}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--first-session", type=int, default=52, help="E2/E3 replay slices start at 52")
    a = ap.parse_args()
    sessions = Sessions(a.replay)
    cal = json.loads(Path(a.calibration).read_text())
    g2, r16 = cal["G2-FP8"], cal["R16-FP8"]

    def tpot_at(unit, c):
        tab = {int(k): v for k, v in unit["tpot_p50_s_by_c"].items()}
        return tab[max(k for k in tab if k <= c)]

    grid = []
    for n in (2, 4, 8):
        for per in (8, 16, 32):
            grid.append(("G2-FP8 replicas", n, per * n, 514048, 64, g2["prefill_tokens_per_s"],
                         g2["ttft_intercept_s"], tpot_at(g2, per)))
    for c in (16, 32, 64, 128):
        grid.append(("R16 DP ranks", 16, c, 539136, 64, r16["prefill_tokens_per_s"],
                     max(r16["ttft_intercept_s"], 0.0), tpot_at(r16, c)))
    policies = {"round_robin": {}, "least_load": {}, "session": {}, "session_lb": {}, "cache_aware": {},
                "cache_aware_abs4": {"abs_thr": 4}, "oracle": {}}
    jobs = []
    for unit, n, c, cap, page, rate, icpt, tpot in grid:
        idx = list(range(a.first_session, a.first_session + 2 * c))
        for pol, kw in policies.items():
            for seed in range(a.seeds):
                jobs.append(((unit, n, c, pol), (idx, c, n, cap, page, rate, icpt, tpot,
                                                 "cache_aware" if pol.startswith("cache_aware") else pol, seed, kw)))
    _G["sessions"] = sessions
    with mp.get_context("fork").Pool(a.workers) as pool:
        outs = pool.map(_one, [j[1] for j in jobs])
    results = []
    for unit, n, c, cap, page, rate, icpt, tpot in grid:
        for pol in policies:
            runs = [o for (key, _), o in zip(jobs, outs) if key == (unit, n, c, pol)]
            row = {"unit": unit, "replicas": n, "concurrency": c, "sessions": 2 * c, "policy": pol,
                   "capacity_tokens": cap, "page": page, "prefill_tok_s": rate, "tpot_s": tpot, "seeds": len(runs)}
            for k in runs[0]:
                vals = [r[k] for r in runs]
                row[k] = statistics.fmean(vals)
                if k == "hit":
                    row["hit_min"], row["hit_max"] = min(vals), max(vals)
            results.append(row)
            print(f'{unit} N={n} C={c} {pol:16s} hit={row["hit"]:.4f} pf={row["prefill_tokens"]:.0f} '
                  f'imb={row["prefill_imbalance"]:.2f} held={row["inflight_tokens_max_over_mean"]:.2f} '
                  f'ttft50={row["ttft_p50_s"]:.2f} tok/s={row["output_tok_per_s"]:.0f}', flush=True)
    bound = []
    for c in (16, 32, 64, 128, 256):
        idx = list(range(a.first_session, a.first_session + 2 * c))
        r = simulate(sessions, idx, c, 1, 1 << 40, 64, 1e9, 0.0, 0.0, "round_robin", 0)
        bound.append({"sessions": len(idx), "concurrency": c, "unbounded_single_cache_hit": r["hit"]})
        print("bound", bound[-1], flush=True)
    Path(a.out).write_text(json.dumps({"grid": results, "unbounded_bound": bound}, indent=1) + "\n")


if __name__ == "__main__":
    main()
