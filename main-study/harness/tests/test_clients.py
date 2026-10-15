import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from aiohttp import web
except ImportError:
    web = None


def fake_app(state):
    async def generate(request):
        body = await request.json()
        state.append(body)
        n = body["sampling_params"]["max_new_tokens"]
        if "input_ids" in body:
            prompt = len(body["input_ids"])
            cached = 4 if prompt > 4 else 0
        else:
            prompt, cached = len(body["text"]), 0
        meta = {"prompt_tokens": prompt, "completion_tokens": n, "cached_tokens": cached, "finish_reason": {"type": "length"}}
        if body.get("return_logprob"):
            start = body["logprob_start_len"]
            k = body["top_logprobs_num"]
            meta["input_token_logprobs"] = [[-0.5, t, None] for t in body["input_ids"][start:]]
            meta["input_top_logprobs"] = [[[-0.1 * j, j, None] for j in range(k)] for _ in body["input_ids"][start:]]
        if body.get("stream"):
            resp = web.StreamResponse()
            await resp.prepare(request)
            for i in range(n):
                chunk = {"text": "x" * (i + 1), "meta_info": {**meta, "completion_tokens": i + 1}}
                await resp.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            await resp.write(b"data: [DONE]\n\n")
            return resp
        return web.json_response({"text": " so the answer is 1,234.", "meta_info": meta})

    app = web.Application()
    app.router.add_post("/generate", generate)
    return app


@unittest.skipIf(web is None, "aiohttp not installed")
class Clients(unittest.TestCase):
    def serve(self, coro_fn):
        async def main():
            state = []
            runner = web.AppRunner(fake_app(state))
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = site._server.sockets[0].getsockname()[1]
            try:
                return await coro_fn(f"http://127.0.0.1:{port}"), state
            finally:
                await runner.cleanup()
        return asyncio.run(main())

    def sessions_file(self, d):
        p = Path(d) / "s.jsonl"
        rows = [{"session": f"s{i}", "seq": list(range(200)), "turns": [{"prompt_len": 10, "answer_len": 5},
                {"prompt_len": 20, "answer_len": 3000}, {"prompt_len": 150, "answer_len": 70}]} for i in range(4)]
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return p

    def test_replay(self):
        import replay_client
        with tempfile.TemporaryDirectory() as d:
            sp = self.sessions_file(d)
            a = SimpleNamespace(sessions=sp, start=1, count=2, concurrency=2, max_answer=8, request_timeout=30, deadline_s=None,
                                out=f"{d}/o.jsonl", summary=f"{d}/s.json")
            rc, sent = self.serve(lambda url: replay_client.run(SimpleNamespace(**vars(a), url=url)))
            recs = [json.loads(x) for x in Path(a.out).read_text().splitlines()]
            self.assertEqual(rc, 0)
            self.assertEqual(len(recs), 6)
            self.assertEqual({r["session"] for r in recs}, {"s1", "s2"})
            self.assertEqual(sorted(len(b["input_ids"]) for b in sent), [10, 10, 20, 20, 150, 150])
            self.assertEqual(sorted(b["sampling_params"]["max_new_tokens"] for b in sent), [5, 5, 8, 8, 8, 8])
            self.assertTrue(all(r["ttft_s"] is not None and r["cached_tokens"] == 4 for r in recs))
            summ = json.loads(Path(a.summary).read_text())
            self.assertEqual((summ["bad_requests"], summ["turns_done"], summ["cut_by_deadline"]), (0, 6, False))

    def test_replay_deadline(self):
        import replay_client
        with tempfile.TemporaryDirectory() as d:
            sp = self.sessions_file(d)
            a = SimpleNamespace(sessions=sp, start=0, count=4, concurrency=2, max_answer=8, request_timeout=30,
                                deadline_s=0.0, out=f"{d}/o.jsonl", summary=f"{d}/s.json")
            rc, sent = self.serve(lambda url: replay_client.run(SimpleNamespace(**vars(a), url=url)))
            summ = json.loads(Path(a.summary).read_text())
            self.assertEqual((rc, summ["turns_done"], summ["cut_by_deadline"]), (0, 0, True))

    def test_gsm8k(self):
        import gsm8k_client
        with tempfile.TemporaryDirectory() as d:
            items = [{"question": f"q{i}", "answer": f"step\n#### {1234 if i % 2 else 7}"} for i in range(6)]
            Path(f"{d}/t.jsonl").write_text("".join(json.dumps(x) + "\n" for x in items))
            a = SimpleNamespace(train=f"{d}/t.jsonl", test=f"{d}/t.jsonl", shots=5, max_new_tokens=16, parallel=4,
                                request_timeout=30, out=f"{d}/o.jsonl", summary=f"{d}/s.json")
            rc, sent = self.serve(lambda url: gsm8k_client.run(SimpleNamespace(**vars(a), url=url)))
            s = json.loads(Path(a.summary).read_text())
            self.assertEqual(rc, 0)
            self.assertEqual(s["n"], 6)
            self.assertAlmostEqual(s["accuracy"], 0.5)
            self.assertTrue(sent[0]["text"].startswith("Question: q0\nAnswer: step"))
            self.assertTrue(sent[0]["text"].endswith("\nAnswer:"))

    def test_logprob(self):
        import logprob_client
        with tempfile.TemporaryDirectory() as d:
            sp = self.sessions_file(d)
            a = SimpleNamespace(sessions=sp, start=0, n=3, forced=64, max_prompt=100, top_k=20, parallel=2,
                                request_timeout=30, out=f"{d}/o.jsonl")
            rc, sent = self.serve(lambda url: logprob_client.run(SimpleNamespace(**vars(a), url=url)))
            recs = [json.loads(x) for x in Path(a.out).read_text().splitlines()]
            self.assertEqual(rc, 0)
            self.assertEqual([r["turn"] for r in recs], [1, 1, 1])
            self.assertTrue(all(len(b["input_ids"]) == 20 + 64 and b["logprob_start_len"] == 20 for b in sent))
            self.assertEqual(len(recs[0]["input_top_logprobs"]), 64)

    def test_pick_turn(self):
        from logprob_client import pick_turn
        turns = [{"prompt_len": 10, "answer_len": 100}, {"prompt_len": 50, "answer_len": 10},
                 {"prompt_len": 90, "answer_len": 64}, {"prompt_len": 200, "answer_len": 500}]
        self.assertEqual(pick_turn(turns, 100, 64), 2)
        self.assertIsNone(pick_turn(turns, 5, 64))


if __name__ == "__main__":
    unittest.main()
