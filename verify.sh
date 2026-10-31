#!/bin/bash
# Check that the one-step series image matches the step-by-step lineage image:
# same files (sha256) under site-packages/vllm and /opt/flash-next, __pycache__ ignored.
#   verify.sh [SERIES_IMAGE] [LINEAGE_IMAGE]
A=${1:-flash-next-vllm-mtp:v031-fp8kv-series-l64}; B=${2:-flash-next-vllm-mtp:v031-fp8kv-tier-l64}
sums() {
  docker run --rm --entrypoint bash "$1" -c 'cd /usr/local/lib/python3.12/dist-packages && find vllm /opt/flash-next -type f -not -path "*/__pycache__/*" -print0 | sort -z | xargs -0 sha256sum'
}
diff <(sums "$A") <(sums "$B") && echo "IDENTICAL: $A == $B ($(sums "$A" | wc -l) files)"
