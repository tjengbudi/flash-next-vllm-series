# Flash-Next v0.31 series up to local-64, in ONE build from the official vLLM v0.31.0 image.
# Same vllm tree as the step-by-step lineage image flash-next-vllm-mtp:v031-fp8kv-tier-l64
# (verify.sh compares them). Order and flags: series; steps: apply.sh; per patch: PATCHES.md.
#
#   docker build -t flash-next-vllm-mtp:v031-fp8kv-series-l64 .
ARG BASE=docker.io/vllm/vllm-openai:v0.31.0@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3
FROM ${BASE}

COPY series apply.sh /tmp/flash-next-series/
COPY patches/ /tmp/flash-next-series/patches/
COPY files/ /tmp/flash-next-series/files/
RUN bash /tmp/flash-next-series/apply.sh && rm -rf /tmp/flash-next-series

# Defaults the lineage images baked in (local-30, local-34, local-35, local-37).
ENV VLLM_CG_DRAFT_META=1
ENV VLLM_PLE_ADVISE_DEDUPE=1 VLLM_PLE_ADVISE_THREADS=3 VLLM_PLE_ADVISE_SPLIT_MIN=16
ENV VLLM_NGRAM_SINGLE_ONLY=1
ENV VLLM_FAST_SAMPLE=1
