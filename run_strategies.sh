#!/usr/bin/env bash
set -euo pipefail

if (( $# == 0 )); then
    echo 'Usage: bash run_strategies.sh CONFIG [CONFIG ...]' >&2
    exit 2
fi
batch_dir=$(mktemp -d "${OUTPUT_PARENT:-.}/runs-XXXXXXXX")
echo "Results: $batch_dir"
pids=()
index=0
for config in "$@"; do
    index=$((index + 1))
    python -m strategysim.run_independent --config "$config" \
        --output-dir "$batch_dir/$index" --concurrency "${CONCURRENCY:-1}" &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
