# Two-minute talk: one MoE model, three platforms, an agentic workload

## Script (about 280 words)

Spoken numbers are rounded; the exact values are in [SUMMARY.md](SUMMARY.md) and the results files.
The script reflects the study as scored on 2026-10-16. The cache-aware follow-up it ends on has since been studied on
CPU only ([SUMMARY.md, next step](SUMMARY.md#next-step-and-cpu-follow-up)).

I served one large mixture-of-experts model, Qwen3 235B, with SGLang on H200 and on GB200.
The workload replays real coding-agent sessions; at low load the prefix-cache hit rate is about 95 percent.
I froze the predictions before each run, and someone else scored the raw data.
Most predictions in the first experiment confirmed directions I had already seen in pilot runs.

First, on the same two GPUs at low load, GB200 FP8 decodes faster than H200 FP8: 7.9 against 9.9 milliseconds per token.
NVFP4 on GB200 is faster again, 6.6 milliseconds, but not lossless.
More answers change than the preregistered limit allows, and it agrees less often with a BF16 reference.

Second, at 32 sessions on two H200s, the KV cache holds only 16 percent of the working set.
Prefix hits fall to 31 percent, and time to first token reaches 27.7 seconds.
That is a capacity limit, not GPU speed.

Third, each platform used its own configuration, and I compared throughput per GPU at the same interactivity.
GB200 FP8 on two GPUs gives 2.0 to 2.2 times H200 on four GPUs; NVFP4 gives 3.3 to 4.3 times.
That is output tokens only, not cost.
Eight H200s with expert parallelism give less per GPU than four, and I don't pin that on one factor.

Fourth, I profiled decode steps with nsys and split GPU time by kernel class.
No single class dominates.
Activation and quantization kernels on H200 and the MoE GEMMs both carry large shares, and the overlap correction is about as large.

Last, one untuned server across four GB200 nodes gave only 2 to 3 percent of one node's throughput per GPU.
It used round-robin dispatch, and its cached prefill fraction was 48 to 61 percent, against about 95 on one node.
I did not isolate the cause, so the next experiment is the same setup with cache-aware routing.

## Likely questions

**Q: So is GB200 twice as fast as H200?**
A: Not as a general claim. That 2.0 to 2.2 is throughput per GPU at matched interactivity, with GB200 on two GPUs and
H200 on four. It counts output tokens only. It is not a cost comparison, and the configurations differ in more than
the GPU.

**Q: What is limiting the H200 at 32 sessions?**
A: KV capacity. The pool was 0.16 of the working set, prefix hits fell to 31.07 percent and TTFT p50 was 27.7
seconds. At that point I am measuring eviction and queueing, not kernel speed.

**Q: Why did eight H200s give less per GPU than four?**
A: I measured that it did: about 0.62 down to 0.28 of the four-GPU value. In decode windows communication carries
about a quarter to a third of the per-token GPU-time difference, and the activation and quantization kernels are the
largest other term. The decode numbers match the serving ratio only at 75 tokens per second per user. I have no
ablation, so I don't name one cause.

**Q: Is NVFP4 safe to use?**
A: GSM8K accuracy was about the same, 95.45 against 95.53 percent. But 30 questions changed answers against FP8, over the
limit of 20, and top-1 agreement with BF16 was 0.946 against 0.970. So I can't call it lossless.

**Q: Why was the 16-GPU server so slow?**
A: I don't know the cause. It was one untuned configuration with round-robin dispatch. The logs, read after the fact and not
preregistered, show much less prefix reuse, many more prefill tokens and more decode log lines without CUDA graphs. Those co-occur with the slowdown,
but nothing isolates which one matters. Cache-aware routing is the experiment that would test the first.
