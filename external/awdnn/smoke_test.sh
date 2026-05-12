#!/bin/bash
# Quick sanity: build the image, then reproduce authors' logs on contracts_re.txt.
# Usage: ./smoke_test.sh [--build]
#   --build  force rebuild of the awdnn:smoke image

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ "${1:-}" == "--build" ]] || ! docker image inspect awdnn:smoke >/dev/null 2>&1; then
    echo "[awdnn smoke] building image …"
    docker build -t awdnn:smoke -f Dockerfile .
fi

OUT=$(mktemp -d -t awdnn-smoke-XXXXXX)
trap 'rm -rf "$OUT"' EXIT

echo "[awdnn smoke] reproduce on upstream contracts_re.txt (authors' log hyperparams)"
docker run --rm -v "$OUT:/work" awdnn:smoke \
    --mode reproduce \
    --train-file /opt/awdnn-src/contracts_re.txt \
    --output /work/reproduce.json \
    --seed 42

echo "[awdnn smoke] result summary:"
python3 -c "
import json, sys
m = json.load(open('$OUT/reproduce.json'))
print(f'  accuracy  : {m[\"accuracy\"]:.4f}')
print(f'  precision : {m[\"precision\"]:.4f}')
print(f'  recall    : {m[\"recall\"]:.4f}')
print(f'  f1        : {m[\"f1\"]:.4f}')
print(f'  n_test    : {m[\"n_test\"]}')
print()
print('Authors shipped logs (evaluations/wdnna/reentrancy/smartoutput_*.log):')
print('  F1 = {0.843, 0.875, 0.867, 0.857, 0.847}, mean 0.858')
print('Paper (Tables 4-5): F1 = 0.950')
"
