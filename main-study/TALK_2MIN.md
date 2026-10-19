# Two-minute talk: one MoE model, three platforms, an agentic workload

## Script (about 280 words)

Spoken numbers are rounded; the exact values are in [SUMMARY.md](SUMMARY.md) and the results files.

I served one large mixture-of-experts model, Qwen3 235B, with SGLang on H200 and on GB200.
The workload replays real coding-agent sessions, so every turn resends a long shared prefix.
I froze each experiment's predictions before running it, and someone else scored the raw data.

First, on the same two GPUs at low load, GB200 FP8 decodes faster than H200 FP8.
The time per output token is 7.9 against 9.9 milliseconds.
NVFP4 on GB200 is faster again, 6.6 milliseconds, but it is not lossless.
More of its answers change than the preregistered limit allows, and it agrees less often with a BF16 reference.

Second, at 32 concurrent sessions on two H200s, the KV cache holds only 16 percent of the working set.
Prefix hits fall to 31 percent, and time to first token reaches 27.7 seconds.
That is a capacity limit, not GPU speed.

Third, I let each platform use its own configuration and compared throughput per GPU at the same interactivity.
GB200 FP8 on two GPUs gives 2.0 to 2.2 times H200 on four GPUs, and NVFP4 gives about 3.3 to 4.3 times.
Going from four H200s to eight with expert parallelism lowers throughput per GPU.

Fourth, I profiled decode steps with nsys to see where the gap goes.
No single kernel class dominates.
Activation and quantization kernels on H200 and the MoE GEMMs both carry large shares, and so does kernel overlap.

Last, one untuned server across four GB200 nodes gave only about 2 to 3 percent of a single node's throughput per GPU.
It used round-robin dispatch, and its cached prefill fraction fell from about 95 percent to between 48 and 61 percent.
I did not isolate the cause.
So the next experiment is the same setup with cache-aware routing.

## Likely questions

**Q: So is GB200 twice as fast as H200?**
A: Not as a general claim. That 2.0 to 2.2 is throughput per GPU at matched interactivity, with GB200 on two GPUs and
H200 on four. It counts output tokens only. It is not a cost comparison, and the configurations differ in more than
the GPU.

**Q: What is limiting the H200 at 32 sessions?**
A: KV capacity. The pool was 0.16 of the working set, prefix hits fell to 31.07 percent and TTFT p50 was 27.7
seconds. At that point I am measuring eviction and queueing, not kernel speed.

**Q: Why did eight H200s give less per GPU than four?**
A: I measured that it did: 0.6158 to 0.2827 of the four-GPU value. In decode windows communication carries 0.2476 to 0.3392 of the
per-token GPU-time difference, and the `other_known` class is the largest other term. I have
no ablation, so I don't name one cause.

**Q: Is NVFP4 safe to use?**
A: GSM8K accuracy was about the same, 95.45 against 95.53 percent. But 30 questions changed answers against FP8, over the
limit of 20, and top-1 agreement with BF16 was 0.946 against 0.970. So I can't call it lossless.

**Q: Why was the 16-GPU server so slow?**
A: I don't know the cause. It was one untuned configuration with round-robin dispatch. The logs show much less
prefix reuse, many more prefill tokens and more decode steps without CUDA graphs. Those co-occur with the slowdown,
but nothing isolates which one matters. Cache-aware routing is the experiment that would test the first.
