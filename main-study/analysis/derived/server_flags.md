| Config | Checkpoint | Flags passed | Harness commit | Jobs (launch) |
|---|---|---|---|---|
| E1 G-FP8 | `Qwen3-235B-A22B-Instruct-2507-FP8` | `--tp 2 --kv-cache-dtype bfloat16` (reconstructed) | `cb9b38c` | 553678 l1, 553983 l2 |
| E1 G-NV4 | `Qwen3-235B-A22B-Instruct-2507-NVFP4` | `--tp 2 --kv-cache-dtype bfloat16` (reconstructed) | `cb9b38c` | 553679 l1, 553773 l2 |
| E1 H-FP8 | `Qwen3-235B-A22B-Instruct-2507-FP8` | `--tp 2 --kv-cache-dtype bfloat16` (reconstructed) | `cb9b38c` | 553686 l1, 553687 l2 |
| E1 REF (BF16) | `Qwen3-235B-A22B-Instruct-2507` | `--tp 4 --kv-cache-dtype bfloat16` (reconstructed) | `cb9b38c` | 554889 |
| E1 H-FP8-TP4 | `Qwen3-235B-A22B-Instruct-2507-FP8` | `--tp 4 --kv-cache-dtype bfloat16` (reconstructed) | `cb9b38c` | 554900 l1, 554910 l2 |
| G2-FP8 | `Qwen3-235B-A22B-Instruct-2507-FP8` | `--tp 2 --kv-cache-dtype bfloat16` (reconstructed) | `d9b86da` | 554224 l1, 554915 l2, 555021 l3 |
| H4-FP8 | `Qwen3-235B-A22B-Instruct-2507-FP8` | `--tp 4 --kv-cache-dtype bfloat16` (reconstructed) | `d9b86da` | 554225 l1, 554270 l2, 554274 l3 |
| G2-NV4 | `Qwen3-235B-A22B-Instruct-2507-NVFP4` | `--tp 2 --kv-cache-dtype bfloat16` (reconstructed) | `d9b86da` | 554273 l1, 554660 l2, 555187 l3 |
| H8-FP8 | `Qwen3-235B-A22B-Instruct-2507-FP8` | `--tp 8 --kv-cache-dtype bfloat16 --ep-size 8` | `2503cec` | 554979 l1, 555000 l2, 555001 l3 |
| H100-8-FP8 | `Qwen3-235B-A22B-Instruct-2507-FP8` | `--tp 8 --kv-cache-dtype bfloat16 --ep-size 8` | `2503cec` | 388635 l1, 388664 l2, 388665 l3 |
| N4-FP8 | `Qwen3-235B-A22B-Instruct-2507-FP8` | `--tp 4 --kv-cache-dtype bfloat16 --ep-size 4` | `2503cec` | 554620 replica1, 554620 replica2, 554620 replica3, 554620 replica4 |
| N4-NV4 | `Qwen3-235B-A22B-Instruct-2507-NVFP4` | `--tp 4 --kv-cache-dtype bfloat16 --ep-size 4` | `2503cec` | 554621 replica1, 554621 replica2, 554621 replica3, 554621 replica4 |
| R16-FP8 | `Qwen3-235B-A22B-Instruct-2507-FP8` | `--tp 16 --kv-cache-dtype bfloat16 --nnodes 4 --attn-dp-size 16 --ep-size 16 --moe-a2a-backend flashinfer --moe-runner-backend flashinfer_trtllm` | `2503cec` | 554984 l1, 555168 l2, 555169 l3 |
| E4 30B TP1 H200 | `Qwen3-30B-A3B-FP8` | `--tp 1 --kv-cache-dtype bfloat16` | `ccebb9c` | 557018 |
| E4 30B TP1 H200 (free) | `Qwen3-30B-A3B-FP8` | `--tp 1 --kv-cache-dtype bfloat16` | `ccebb9c` | 557018 |
| E4 30B TP2 GB200 | `Qwen3-30B-A3B-FP8` | `--tp 2 --kv-cache-dtype bfloat16` | `ccebb9c` | 557081 |
| E4 30B TP2 GB200 (free) | `Qwen3-30B-A3B-FP8` | `--tp 2 --kv-cache-dtype bfloat16` | `ccebb9c` | 557081 |
| E4 G2-FP8 | `Qwen3-235B-A22B-Instruct-2507-FP8` | `--tp 2 --kv-cache-dtype bfloat16` | `ccebb9c` | 557081 |
| E4 G2-NV4 | `Qwen3-235B-A22B-Instruct-2507-NVFP4` | `--tp 2 --kv-cache-dtype bfloat16` | `ccebb9c` | 557081 |
| E4 30B TP2 H200 | `Qwen3-30B-A3B-FP8` | `--tp 2 --kv-cache-dtype bfloat16` | `ccebb9c` | 557426 |
| E4 30B TP2 H200 (free) | `Qwen3-30B-A3B-FP8` | `--tp 2 --kv-cache-dtype bfloat16` | `ccebb9c` | 557426 |
| E4 H4-FP8 | `Qwen3-235B-A22B-Instruct-2507-FP8` | `--tp 4 --kv-cache-dtype bfloat16` | `ccebb9c` | 557426 |
| E4 H8-FP8 | `Qwen3-235B-A22B-Instruct-2507-FP8` | `--tp 8 --kv-cache-dtype bfloat16 --ep-size 8` | `ccebb9c` | 557426 |

| Config | tp | ep | attn-dp | nnodes | quant | attention | moe runner | moe a2a | chunked prefill | mem frac | page | dispatch | KV heads / GPU |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 G-FP8 | 2 | 1 | 1 | 1 | fp8 | trtllm_mha | flashinfer_trtllm | none | 16384 | 0.858 | 64 | round_robin | 2.0 |
| E1 G-NV4 | 2 | 1 | 1 | 1 | modelopt_fp4 | trtllm_mha | flashinfer_trtllm | none | 16384 | 0.858 | 64 | round_robin | 2.0 |
| E1 H-FP8 | 2 | 1 | 1 | 1 | None | fa3 | auto | none | 8192 | 0.902 | 1 | round_robin | 2.0 |
| E1 REF (BF16) | 4 | 1 | 1 | 1 | None | fa3 | auto | none | 8192 | 0.897 | 1 | round_robin | 1.0 |
| E1 H-FP8-TP4 | 4 | 1 | 1 | 1 | None | fa3 | auto | none | 8192 | 0.897 | 1 | round_robin | 1.0 |
| G2-FP8 | 2 | 1 | 1 | 1 | fp8 | trtllm_mha | flashinfer_trtllm | none | 16384 | 0.858 | 64 | round_robin | 2.0 |
| H4-FP8 | 4 | 1 | 1 | 1 | None | fa3 | auto | none | 8192 | 0.897 | 1 | round_robin | 1.0 |
| G2-NV4 | 2 | 1 | 1 | 1 | modelopt_fp4 | trtllm_mha | flashinfer_trtllm | none | 16384 | 0.858 | 64 | round_robin | 2.0 |
| H8-FP8 | 8 | 8 | 1 | 1 | None | fa3 | auto | none | 8192 | 0.893 | 1 | round_robin | 1.0 |
| H100-8-FP8 | 8 | 8 | 1 | 1 | None | fa3 | auto | none | 8192 | 0.812 | 1 | round_robin | 1.0 |
| N4-FP8 | 4 | 4 | 1 | 1 | fp8 | trtllm_mha | flashinfer_trtllm | none | 16384 | 0.856 | 64 | round_robin | 1.0 |
| N4-NV4 | 4 | 4 | 1 | 1 | modelopt_fp4 | trtllm_mha | flashinfer_trtllm | none | 16384 | 0.856 | 64 | round_robin | 1.0 |
| R16-FP8 | 16 | 16 | 16 | 4 | fp8 | trtllm_mha | flashinfer_trtllm | flashinfer | 1024 | 0.654 | 64 | round_robin | 4.0 |
| E4 30B TP1 H200 | 1 | 1 | 1 | 1 | None | fa3 | auto | none | 8192 | 0.903 | 1 | round_robin | — |
| E4 30B TP1 H200 (free) | 1 | 1 | 1 | 1 | None | fa3 | auto | none | 8192 | 0.903 | 1 | round_robin | — |
| E4 30B TP2 GB200 | 2 | 1 | 1 | 1 | fp8 | trtllm_mha | flashinfer_trtllm | none | 16384 | 0.858 | 64 | round_robin | — |
| E4 30B TP2 GB200 (free) | 2 | 1 | 1 | 1 | fp8 | trtllm_mha | flashinfer_trtllm | none | 16384 | 0.858 | 64 | round_robin | — |
| E4 G2-FP8 | 2 | 1 | 1 | 1 | fp8 | trtllm_mha | flashinfer_trtllm | none | 16384 | 0.858 | 64 | round_robin | 2.0 |
| E4 G2-NV4 | 2 | 1 | 1 | 1 | modelopt_fp4 | trtllm_mha | flashinfer_trtllm | none | 16384 | 0.858 | 64 | round_robin | 2.0 |
| E4 30B TP2 H200 | 2 | 1 | 1 | 1 | None | fa3 | auto | none | 8192 | 0.902 | 1 | round_robin | — |
| E4 30B TP2 H200 (free) | 2 | 1 | 1 | 1 | None | fa3 | auto | none | 8192 | 0.902 | 1 | round_robin | — |
| E4 H4-FP8 | 4 | 1 | 1 | 1 | None | fa3 | auto | none | 8192 | 0.897 | 1 | round_robin | 1.0 |
| E4 H8-FP8 | 8 | 8 | 1 | 1 | None | fa3 | auto | none | 8192 | 0.893 | 1 | round_robin | 1.0 |
