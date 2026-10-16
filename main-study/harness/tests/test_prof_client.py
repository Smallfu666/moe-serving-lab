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
    import prof_client
except ImportError:
    web = None


def fake_app(log_path, state, stop_after_s=0.3, stop_line=True, token_s=0.005):
    async def generate(request):
        body = await request.json()
        state["generate"].append(body)
        resp = web.StreamResponse()
        await resp.prepare(request)
        for i in range(body["sampling_params"]["max_new_tokens"]):
            await asyncio.sleep(token_s)
            await resp.write(b"data: " + json.dumps({"meta_info": {"completion_tokens": i + 1}}).encode() + b"\n\n")
        await resp.write(b"data: [DONE]\n\n")
        return resp

    async def start_profile(request):
        state["profile"].append(await request.json())

        async def later():
            await asyncio.sleep(stop_after_s)
            if stop_line:
                with open(log_path, "a") as f:
                    f.write("[x] Stop profiling...\n")
        asyncio.create_task(later())
        return web.Response(text="Start profiling.")

    async def flush(request):
        return web.Response(text="ok")

    app = web.Application()
    app.router.add_post("/generate", generate)
    app.router.add_post("/start_profile", start_profile)
    app.router.add_post("/flush_cache", flush)
    return app


@unittest.skipIf(web is None, "aiohttp not installed")
class ProfClient(unittest.TestCase):
    def run_mode(self, mode, sessions, **kw):
        async def main(d):
            state = {"generate": [], "profile": []}
            log = d + "/server.log"
            open(log, "w").write("[x] Stop profiling earlier run\n")
            runner = web.AppRunner(fake_app(log, state, **{k: kw.pop(k) for k in list(kw) if k in
                                                         ("stop_after_s", "stop_line", "token_s")}))
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = site._server.sockets[0].getsockname()[1]
            a = SimpleNamespace(url=f"http://127.0.0.1:{port}", server_log=log, tag="t", steps=8, window_timeout=2,
                                sessions=sessions, start=0, batch=2, prompt_len=3, warm=4, max_new=200,
                                warm_timeout=5, request_timeout=30, delay=0.1)
            a.__dict__.update(kw)
            try:
                fn = prof_client.run_decode if mode == "decode" else prof_client.run_trigger
                return await fn(a), state
            finally:
                await runner.cleanup()
        with tempfile.TemporaryDirectory() as d:
            return asyncio.run(main(d))

    def sessions(self, d, lens):
        p = d + "/s.jsonl"
        with open(p, "w") as f:
            for n in lens:
                f.write(json.dumps({"seq": list(range(n))}) + "\n")
        return p

    def test_decode_window_ok(self):
        with tempfile.TemporaryDirectory() as d:
            s, state = self.run_mode("decode", self.sessions(d, [2, 5, 9, 4]))
        self.assertEqual(s["rc"], 0)
        self.assertEqual(s["session_rows"], [1, 2])
        self.assertTrue(all(t >= 4 for t in s["tokens_at_start"]))
        self.assertEqual(state["profile"][0]["activities"], ["CUDA_PROFILER"])
        self.assertEqual(state["profile"][0]["num_steps"], 8)
        self.assertEqual([len(g["input_ids"]) for g in state["generate"]], [3, 3])

    def test_too_few_prompts(self):
        with tempfile.TemporaryDirectory() as d:
            s, _ = self.run_mode("decode", self.sessions(d, [2, 5, 1]))
        self.assertEqual(s["rc"], 2)

    def test_requests_finish_before_stop(self):
        with tempfile.TemporaryDirectory() as d:
            s, _ = self.run_mode("decode", self.sessions(d, [5, 5]), max_new=30, stop_after_s=1.0)
        self.assertEqual(s["rc"], 3)

    def test_stop_line_missing(self):
        with tempfile.TemporaryDirectory() as d:
            s, _ = self.run_mode("decode", self.sessions(d, [5, 5]), stop_line=False, max_new=600)
        self.assertEqual(s["rc"], 5)

    def test_trigger(self):
        s, state = self.run_mode("trigger", None)
        self.assertEqual(s["rc"], 0)
        self.assertEqual(len(state["profile"]), 1)


if __name__ == "__main__":
    unittest.main()
