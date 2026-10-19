# What I learned: prompts

Questions only. Answer each in your own words, from the results files and the code you read, before writing the
"What I learned" sections in `results/e1-results.md` … `e4-results.md`. If you cannot answer one without guessing,
write down what you would need to measure.

1. On 2 × H200 at c = 32, the KV pool is 0.16 of the working set. Walk through what the server does when a new turn
   arrives and its prefix has been evicted. Why does that show up mostly in TTFT, and why is it not a statement about
   GPU speed?
2. What do the results let you say about why throughput per GPU falls when H200 goes from TP4 to TP8 + EP8, and
   what does E4's "partial" F-COMM verdict rule out? What would you need before naming a cause?
3. In E4, which kernels make up most of `other_known` on H200 at each B, and what would you need to measure to
   explain their share? What GB200 kernel sits in a different class, and why does that matter for the comparison?
4. The overlap term in E4 is about as large as the biggest class share. What does "sum basis" mean, why can the class
   shares add up to more than 1, and how would a union-basis view change the story?
5. F-BRIDGE fires for GB200 NVFP4 vs H200 TP4 at 100 tok/s/user. What does it mean that decode steps alone do not
   account for the E2 ratio, and which parts of serving lie outside decode steps?
6. NVFP4 matches FP8 on GSM8K accuracy but moves 30 answers and agrees less with BF16. Why is equal accuracy not
   evidence of lossless quantization, and which check would you trust more?
7. The 16-GPU server ran TP16 + attention-DP16 + EP16 with round-robin dispatch. Explain how request dispatch
   interacts with a per-rank prefix cache. Then design the cache-aware follow-up: what stays fixed, what you predict,
   and what result would prove you wrong.
8. Why is 4 GB200 GPUs ÷ 2 GB200 GPUs (0.8503 / 0.7974 / 0.7078) not a GPU-count scaling result? List every factor
   that differs between those two configurations.
