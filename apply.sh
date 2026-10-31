#!/bin/bash
# Apply the whole Flash-Next series to a vLLM v0.31.0 site-packages tree, in order.
#   apply.sh [SITE_PACKAGES]   (default /usr/local/lib/python3.12/dist-packages)
# Reproduces image flash-next-vllm-mtp:v031-fp8kv-tier-l64 from vllm/vllm-openai:v0.31.0
# (plus the ENV lines in the Dockerfile next to this script).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SITE=${1:-/usr/local/lib/python3.12/dist-packages}
cd "$SITE"
while read -r name flag _; do
  case "$name" in ''|'#'*) continue ;; esac
  case "$name" in
    *local-18-mtp-draft-vocab.patch)
      # The draft vocab id lists the MTP head loads (local-18; VLLM_MTP_DRAFT_VOCAB picks one).
      cp "$HERE"/files/mtp_draft_vocab_ids*.json vllm/models/qwen4_exp/nvidia/ ;;
    *local-35-ngram-lookup-single-request.patch)
      # The n-gram GPU speculator package the patch fills in.
      mkdir -p vllm/v1/worker/gpu/spec_decode/ngram
      touch vllm/v1/worker/gpu/spec_decode/ngram/__init__.py ;;
    *local-47-decisions-main.patch)
      rm -rf vllm/entrypoints/generate/structured_decisions/__pycache__ ;;
  esac
  opts=(-p1 --batch --forward --fuzz=0)
  [ "$flag" = E ] && opts+=(-E)
  echo "applying $name"
  patch "${opts[@]}" < "$HERE/patches/$name" > /dev/null
  case "$name" in
    *local-47-decisions-main.patch)
      # local-47 replaces the old strategy module; it must be gone.
      test ! -e vllm/entrypoints/generate/structured_decisions/strategies.py ;;
  esac
done < "$HERE/series"
# The INT4 MTP head builder the base image ships (flash-next-v031-mtp-fp8kv).
mkdir -p /opt/flash-next && cp "$HERE/files/make_mtp_int4.py" /opt/flash-next/
# Byte-compile what the series touched, like the per-step Dockerfiles did.
grep -h '^+++ b/' "$HERE"/patches/*.patch | sed 's|^+++ b/||; s|\t.*||' | grep '\.py$' | sort -u \
  | while read -r f; do [ -e "$f" ] && echo "$f"; done | xargs python3 -m py_compile
echo "series applied: $(grep -c '^[0-9]' "$HERE/series") patches"
