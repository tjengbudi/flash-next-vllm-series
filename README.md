# Qwen3.8-Flash-Next on vLLM v0.31.0: a 60-patch series for 3 × 24 GB GPUs

This series takes the official `vllm/vllm-openai:v0.31.0` image to the build that serves
[Swift 1.5 Qwen3.8-Flash-Next Uncensored (INT4 mixed, MTP)](https://huggingface.co/tjengbudi/Swift1.5-Qwen3.8-Flash-Next-Uncensored-INT4-Mixed-AutoRound-MTP)
on three RTX 3090s. It applies in one Docker build, and the result is file-for-file identical to
the image that was built and measured step by step.

- **001–015** are Minachist's Flash-Next patches (Qwen4Exp port, PLE, QSA staged prefill, MTP),
  regenerated for v0.31.0.
- **016–060** are our patches on top. Some port open or merged upstream vLLM PRs.
  [PATCHES.md](PATCHES.md) has one row per patch, with its source.

This is unofficial work, not reviewed or endorsed by the vLLM project, Qwen, UkisAI or Minachist.

## What the series adds (highlights)

| Area | Patches |
|---|---|
| FP8 E4M3 KV for QSA on sm_86 (Ampere), BF16 dots | local-02 |
| Host-RAM KV offload tier that carries the QSA host K/V; tier fixes | local-08, -11, -41, -43, -62 |
| INT8 PLE n-gram table locked in RAM (52 GB instead of 100 GB, same NLL) | local-49 |
| Pipeline-parallel decode: decodes spread over PP microbatches, packed broadcast, sync fixes | local-12, -20, -59 |
| MTP: draft-only vocabulary head, INT8 dense linears, CUDA-graph verify / drafter metadata | local-18, -30, -37, -53 |
| GPU n-gram lookup before MTP (single request) | local-35 |
| RecoverSSM: 57 → 19 Mamba blocks per request | local-42, -44, -45 |
| Prefill scheduling: cadence / burst so long prefills do not starve other streams | local-48b, -48c |
| QSA prefill: copy-engine staging, bucket partition, sm_86 launch config | local-07, -50, -51 |
| Structured decisions API (`/v1/decisions`, `/v1/systemone`), two-letter labels | local-09, -46, -47, -64 |
| Boot / memory: chunked Marlin MoE repack, M-RoPE cache 1×, warm-up sizes, name prefilter | local-06, -10, -52, -61 |

## Build

```bash
git clone https://github.com/tjengbudi/flash-next-vllm-series
cd flash-next-vllm-series
docker build -t flash-next-vllm-mtp:v031-fp8kv-series-l64 .
```

The `Dockerfile` starts from `vllm/vllm-openai:v0.31.0` (pinned by digest) and runs `apply.sh`.
`apply.sh` applies `series` in order and takes care of three extra steps (see PATCHES.md).
`full-v0.31.0-to-l64.diff` holds the same changes as one diff, for reading.

## Run

```bash
# the model repo is gated: accept the terms on its page, then `hf auth login`
hf download tjengbudi/Swift1.5-Qwen3.8-Flash-Next-Uncensored-INT4-Mixed-AutoRound-MTP --local-dir /models/swift-uncensored
MODEL_DIR=/models/swift-uncensored docker compose -f examples/compose.yml up -d
curl http://127.0.0.1:8000/v1/models
```

`examples/compose.yml` sets every knob, and its comments carry the measurement log behind
each value. Most patches do nothing until their env knob is on.

**Tested only on:** 3 × RTX 3090 (24 GB), PCIe without P2P, inside a VM, 147 GiB host RAM.
That means pipeline parallel 3 (layers `16,17,15`), MTP n=2, FP8 KV, 262,144-token context,
and up to 3 sequences.

Host RAM it needs: about 48 GiB for the INT8 PLE table, plus `KV_OFFLOAD_GB` (default 40),
plus the QSA host K/V pool. Tensor parallelism is not supported by these patches.

Measured on that rig with club-3090 `bench.sh`, single stream:

| | |
|---|---|
| Decode, essay / code | 138.8 / 169.2 tok/s |
| Prefill, 10K / 90K | ~3,900 / ~3,200 tok/s |
| 3 streams | ~300 tok/s total |

## Credits

- **[Minachist](https://huggingface.co/Minachist)**: the Flash-Next vLLM patches (001–015),
  the INT4/INT6 AutoRound quantizations, and `make_mtp_int4.py`. Minachist asks that upstream
  merges keep them as commit author, or add
  `Co-authored-by: Minachist <84688797+aiueonicked@users.noreply.github.com>`.
- **[vLLM](https://github.com/vllm-project/vllm)** (Apache-2.0): the engine. Credit also goes
  to the contributors of the upstream PRs this series ports (PATCHES.md, NOTICE). That
  includes #55506 by Karl0007, whose root cause zebgop-ops found first.
- **[club-3090](https://github.com/noonghunna/club-3090)**: the serving stack, compose layout
  and bench / verify / quality tooling this series was built, tested and measured with.
- **[roozich/Qwen3.8-Flash-Spark](https://github.com/roozich/Qwen3.8-Flash-Spark)**
  (Apache-2.0): the MTP weight-name prefilter in local-61.
- **[Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** and
  **[UkisAI](https://huggingface.co/ukisai/Swift1.5-Qwen3.8-Flash-Next)**: the model and the
  Swift 1.5 fine-tune.

## License

Apache-2.0 ([LICENSE](LICENSE)). `files/make_mtp_int4.py` is Minachist's and is MIT
([LICENSE.MIT](LICENSE.MIT)). [NOTICE](NOTICE) has the attributions. Model weights are not
part of this repository and carry their own licences.
