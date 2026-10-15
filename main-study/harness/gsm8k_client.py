"""GSM8K, 5-shot, greedy, against SGLang's /generate; one JSON line per question for paired-flip scoring.

Prompt format and answer extraction follow `sglang.test.few_shot_gsm8k` (shots are "Question: ...\\nAnswer: ..."
blocks; the prediction is the last integer in the answer). Shots come from the train split, so all 1,319 test
questions are scored.
"""

import argparse
import asyncio
import json
import re
from pathlib import Path

import aiohttp

INVALID = None
STOP = ["Question", "Assistant:", "<|separator|>"]


def answer_value(text):
    numbers = re.findall(r"\d+", text.replace(",", ""))
    return int(numbers[-1]) if numbers else INVALID


def example(item, with_answer):
    s = "Question: " + item["question"] + "\nAnswer:"
    return s + " " + item["answer"] if with_answer else s


def build_prompts(train, test, shots):
    prefix = "".join(example(train[i], True) + "\n\n" for i in range(shots))
    return [prefix + example(q, False) for q in test]


async def run(a):
    train = [json.loads(x) for x in Path(a.train).read_text().splitlines() if x.strip()]
    test = [json.loads(x) for x in Path(a.test).read_text().splitlines() if x.strip()]
    prompts = build_prompts(train, test, a.shots)
    sem = asyncio.Semaphore(a.parallel)
    results = [None] * len(test)

    async def one(i, http):
        payload = {"text": prompts[i], "sampling_params": {"max_new_tokens": a.max_new_tokens, "temperature": 0.0,
                                                             "stop": STOP}}
        async with sem:
            try:
                async with http.post(a.url + "/generate", json=payload,
                                     timeout=aiohttp.ClientTimeout(total=a.request_timeout)) as r:
                    body = await r.json() if r.status == 200 else {"error": f"http {r.status}", "error_kind": "server"}
            except asyncio.TimeoutError:
                body = {"error": "TimeoutError", "error_kind": "client_timeout"}
            except aiohttp.ClientError as e:
                body = {"error": type(e).__name__, "error_kind": "client_exception"}
        text = body.get("text", "")
        meta = body.get("meta_info", {})
        pred, label = answer_value(text), answer_value(test[i]["answer"].split("####")[-1])
        results[i] = {"idx": i, "pred": pred, "label": label, "correct": pred is not None and pred == label,
                      "invalid": pred is None, "finish_reason": meta.get("finish_reason"),
                      "completion_tokens": meta.get("completion_tokens"), "error": body.get("error"),
                  "error_kind": body.get("error_kind")}

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=a.parallel)) as http:
        await asyncio.gather(*(one(i, http) for i in range(len(test))))
    with open(a.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    errors = sum(r["error"] is not None for r in results)
    summary = {"n": len(results), "accuracy": sum(r["correct"] for r in results) / len(results),
               "invalid": sum(r["invalid"] for r in results) / len(results), "errors": errors}
    Path(a.summary).write_text(json.dumps(summary) + "\n")
    print(json.dumps(summary))
    return 0 if errors == 0 else 3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--parallel", type=int, default=128)
    ap.add_argument("--request-timeout", type=float, default=900)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    raise SystemExit(asyncio.run(run(ap.parse_args())))


if __name__ == "__main__":
    main()
