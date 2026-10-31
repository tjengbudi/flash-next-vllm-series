# Flash-Next on vLLM v0.31.0: the whole patch series up to local-64

Every change between the official `vllm/vllm-openai:v0.31.0` image and the Swift Flash-Next
image `flash-next-vllm-mtp:v031-fp8kv-tier-l64` (2026-10-30), in one place. That image was
built step by step on the maintainer's rig (one folder per patch, the `series` file names them);
this repository holds the same patches in build order.

| File | What it is |
|---|---|
| `patches/NNN-*.patch` | the 60 patches, numbered in the order they apply |
| `series` | apply order, the `-E` flag where the build used it, and the folder each patch came from |
| `apply.sh` | applies the series to a site-packages tree, with the three extra steps below |
| `files/` | `make_mtp_int4.py` (to `/opt/flash-next/`), the MTP draft-vocab id lists (local-18), and `make_ple_int8.py` (builds the INT8 PLE table local-49 maps from the BF16 n-gram shards; byte-identical rows to QEngine's file, checked 2026-10-31; not copied into the image) |
| `Dockerfile` | one build from the official image: `docker build -t flash-next-vllm-mtp:v031-fp8kv-series-l64 .` |
| `verify.sh` | compares two images file by file (sha256); used to prove the one-step build equals the step-by-step one |
| `full-v0.31.0-to-l64.diff` | the same result as ONE diff of `site-packages/vllm` (1.9 MB, 100 files), for reading |

Checked 2026-10-30: the image built from this `Dockerfile` has the same 5,220 files under
`site-packages/vllm` and `/opt/flash-next` as `-tier-l64` (`verify.sh`: IDENTICAL), and the
same ENV. `full-v0.31.0-to-l64.diff` applied to a clean v0.31.0 tree gives the same tree
except the empty `vllm/v1/worker/gpu/spec_decode/ngram/__init__.py`, which a diff cannot
create; create it by hand if you use the single diff.

Base image: `docker.io/vllm/vllm-openai:v0.31.0@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3`.

Extra steps in `apply.sh` (the lineage Dockerfiles did them in their RUN lines):

- before 030 (local-18): copy `files/mtp_draft_vocab_ids*.json` into `vllm/models/qwen4_exp/nvidia/`
- before 038 (local-35): create the empty package `vllm/v1/worker/gpu/spec_decode/ngram/`
- before 046 (local-47): remove `structured_decisions/__pycache__`; after it, `strategies.py` must be gone

ENV baked into the image (lineage defaults): `VLLM_CG_DRAFT_META=1`,
`VLLM_PLE_ADVISE_DEDUPE=1 VLLM_PLE_ADVISE_THREADS=3 VLLM_PLE_ADVISE_SPLIT_MIN=16`,
`VLLM_NGRAM_SINGLE_ONLY=1`, `VLLM_FAST_SAMPLE=1`. Every other knob is set by the compose
(`examples/compose.yml`); most patches are inert
until their env knob is on.

Known leftovers, kept so the tree stays identical to `-tier-l64`: a few `*.py.orig` files
(patch backups that two diffs carried in, e.g. `vllm/config/scheduler.py.orig`). Python never
imports them.

## The patches

| # | Patch | What it does | Source |
|---|---|---|---|
| 001 | flash-next-vllm | Qwen3.8 Flash-Next (Qwen4Exp) on vLLM 0.31: hyper-connections, NVIDIA PLE module, QSA staged prefill; port notes at the top | Flash-Next author patch, ported |
| 002 | decode-01-ple-host-gather | PLE rows gathered on the host for decode | Flash-Next |
| 003 | decode-02-model-state-hook | model-state hook the decode patches use | Flash-Next |
| 004 | ttft-01-ple-page-prefetch | prefetch PLE pages for prefill (TTFT) | Flash-Next |
| 005 | decode-03-pp-deferred-recv | deferred PP receive | Flash-Next |
| 006 | decode-04-hc-fused-int8 | fused INT8 hyper-connection kernels, 1-4 tokens | Flash-Next |
| 007 | decode-05-moe-fused | fused MoE decode, small token counts | Flash-Next |
| 008 | decode-06-ple-gather-advise | madvise for the PLE rows of the next step | Flash-Next |
| 009 | decode-07-qsa-row-cache | QSA decode row cache | Flash-Next |
| 010 | mem-01-pp-embed-head | embedding / head placement under PP (VRAM) | Flash-Next |
| 011 | vision-01-streamed-tower | streamed vision tower | Flash-Next |
| 012 | upstream-01-mamba-align-req-slot | mamba align-mode request slot | vLLM #55506 (open) |
| 013 | mtp-01-enable | MTP drafter for Qwen4Exp | Flash-Next |
| 014 | mtp-02-three-cards | MTP with PP=3 | Flash-Next |
| 015 | mtp-03-structured-output-drafts | drafts with structured output | Flash-Next |
| 016 | local-01 ple-finalize-after-all-rows | PLE finalize after every row is gathered | own |
| 017 | local-02 qsa-fp8-kv | FP8 E4M3 QSA K/V on sm_86, BF16 dots | on vLLM #55557 |
| 018 | local-03 decode-cpu | per-token CPU cuts on PP0 | own |
| 019 | local-04 qsa-host-kv-cow | copy-on-write for QSA host K/V on prefix hits | own |
| 020 | local-06 moe-marlin-repack-chunked | Marlin MoE repack in chunks (0.31 load OOM) | own |
| 021 | local-07 qsa-staged-sm86-config | staged prefill launch config for sm_86 | own |
| 022 | local-08 qsa-host-kv-offload-tier | CPU KV offload tier carries the QSA host K/V | own, on OffloadingConnector |
| 023 | local-09 structured-decisions | `/v1/systemone` backported from 0.32 | vLLM #59299 and follow-ups |
| 024 | local-10 mrope-cache-1x | M-RoPE cache at 1x positions (-96 MiB per GPU) | vLLM #59853 |
| 025 | local-11 spec-decode-reservation | spec-decode blocks reserved per KV group for async loads | vLLM #59880 |
| 026 | local-12 pp-decode-spread | decodes spread over PP microbatches | own |
| 027 | local-14 gdn-spec-slice | GDN spec-decode slicing | own |
| 028 | local-15 sampler-topp-bundle | sampler top-p bundle | see the patch |
| 029 | local-21 attn-metadata-cg | attention metadata in CUDA graphs | own |
| 030 | local-18 mtp-draft-vocab | draft-only vocabulary head for the MTP drafter (+ id lists) | own |
| 031 | local-20 pp-packed-broadcast | packed PP broadcast | own |
| 032 | local-21c marlin-moe-det-align | deterministic Marlin MoE alignment | own |
| 033 | local-21d ...-prefill-only | the same, prefill only | own |
| 034 | local-21e ...-triton | the same, Triton align | own |
| 035 | local-29 ple-ctx-first-rank-only | PLE n-gram context on PP rank 0 only | own |
| 036 | local-30 draft-attn-metadata-cg | drafter attention metadata in CUDA graphs | own |
| 037 | local-34 ple-advise-dedupe-threads | PLE advise dedupe + 3 threads | own |
| 038 | local-35 ngram-lookup-single-request | GPU n-gram lookup before MTP, only while 1 request runs | vLLM #60615 / #40704 + own |
| 039 | local-37 fast-sample-verify-cg | decode verify in a CUDA graph on PP2 | own |
| 040 | local-41 partial-tail-chunk-store | tier stores the partial tail chunk | own |
| 041 | local-42 recoverssm | RecoverSSM: 57 -> 19 Mamba blocks per request | vLLM #58863 |
| 042 | local-43 cow-pairwise | pairwise CoW (OOM fix) | own |
| 043 | local-44 recoverssm-pp-cg | RecoverSSM with PP + CUDA graphs | own |
| 044 | local-45 recoverssm-meta-reuse | GDN metadata reuse | own |
| 045 | local-46 decisions-fixes | decisions with MTP drafts; hybrid shared-prefix warm-up | vLLM #61219, #61192 |
| 046 | local-47 decisions-main | decisions re-based on upstream main + 8 open PRs, `/v1/decisions` | vLLM main c32575477 + PRs |
| 047 | local-48b prefill-cadence-burst | prefill schedule interval + burst | vLLM #54627 + own |
| 048 | local-48c prefill-cadence-short | short prompts skip the cadence hold | own |
| 049 | local-49 ple-int8 | INT8 PLE table locked in RAM | own |
| 050 | local-50 qsa-ce-copy-own-rowcache | QSA copy-engine staging, own row cache | own |
| 051 | local-51 qsa-bucket-partition | QSA bucket partition | own |
| 052 | local-52 extra-warmup-sizes | warm every linear at the listed sizes (lazy module OOM) | own |
| 053 | local-53 skinny-mtpint8-compacttopk | skinny GEMM, INT8 MTP dense linears, compact top-k | own |
| 054 | local-54 skinny-verify | diagnostic skinny-vs-normal check (off) | own |
| 055 | local-55 skinny-skip-routers | MoE routers stay on the normal GEMM | own |
| 056 | local-56 skinny-thin-only | skinny bf16 only for thin weights (block_inject) | own |
| 057 | local-59 ppsync-ba-overlap | PP draft filter without sync; in_proj_ba on the aux stream; CoW log at debug | vLLM #64458, #64425 |
| 058 | local-61 boot-prefilter-thp | MTP weight-name prefilter; `VLLM_QSA_HOST_THP` knob | roozich prefilter + own |
| 059 | local-62 inflight-kv-loads | cap on requests holding blocks for tier loads (scheduler stall fix) | vLLM #64628 + env fallback |
| 060 | local-64 pair-labels | decision choice labels past Z (AA, AB, ...), up to 255 options | as SGLang decisions |

Numbers missing from the table: 05 was folded into 02 in the 0.31 port, 48 is inside 48b, and
13, 16, 17, 19, 22-28, 31-33, 36, 38-40, 57, 58, 60 and 63 were rejected or superseded, so they
are not in `-tier-l64`.

## Credits

- **Minachist**: the Flash-Next vLLM patches (001-015: the Qwen4Exp / Flash-Next port, PLE,
  QSA staged prefill, MTP) and the base quant
  `Minachist/Swift1.5-Qwen3.8-Flash-Next-INT4-Mixed-AutoRound`. If any of this goes
  upstream, Minachist asks to stay the commit author or to get the trailer
  `Co-authored-by: Minachist <84688797+aiueonicked@users.noreply.github.com>` (patch 001 header).
- **club-3090** (https://github.com/noonghunna/club-3090): the serving stack, compose layout and
  bench / verify / quality tooling this series was built and tested with.
- **vLLM** (https://github.com/vllm-project/vllm, Apache-2.0): the engine. The series applies to
  `vllm/vllm-openai:v0.31.0`, and several patches are ports of upstream vLLM PRs (the
  Source column above names them).
