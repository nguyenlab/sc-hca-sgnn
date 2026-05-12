#!/usr/bin/env bash
# Smoke-test gate for Step 1 — runs ReEP against 3 known-positive SmartBugs
# reentrancy contracts. If ReEP fails to flag them, STOP before attempting
# the full dataset run.
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-reep:smoke}"
CONTRACTS_DIR="${CONTRACTS_DIR:-/tmp/reep-inspect/ReEP/Datasets/DB2/smartbugs_dataset/reentrancy}"
OUT_DIR="$(mktemp -d)"
trap 'rm -rf "$OUT_DIR"' EXIT

SAMPLES=(
  simple_dao.sol
  reentrancy_bonus.sol
  reentrancy_dao.sol
)

echo "== Building image ${IMAGE_TAG} =="
docker build -t "${IMAGE_TAG}" -f "$(dirname "$0")/Dockerfile" "$(dirname "$0")"

echo
echo "== Running smoke test on ${#SAMPLES[@]} known-positive contracts =="
FAIL=0
for sample in "${SAMPLES[@]}"; do
  src="${CONTRACTS_DIR}/${sample}"
  out="${OUT_DIR}/${sample}.csv"
  if [[ ! -f "$src" ]]; then
    echo "MISSING  ${sample}  (expected at $src)"
    FAIL=$((FAIL+1))
    continue
  fi
  cp "$src" "${OUT_DIR}/${sample}"

  # -a auto-mode picks the first reentrancy candidate Slither finds.
  timeout 180 docker run --rm \
    -v "${OUT_DIR}:/work" \
    "${IMAGE_TAG}" \
    -f "/work/${sample}" -r -a -o "/work/${sample}.csv" \
    > "${OUT_DIR}/${sample}.stdout" 2>&1 || true

  if [[ ! -s "$out" ]]; then
    echo "NO_OUTPUT  ${sample}"
    FAIL=$((FAIL+1))
    continue
  fi

  # CSV column 2 is is_bug; flag as pass if any row reports True.
  if grep -q ',True,' "$out"; then
    echo "PASS       ${sample}"
  else
    echo "MISS       ${sample}  (ReEP did not flag reentrancy)"
    FAIL=$((FAIL+1))
  fi
done

echo
if (( FAIL > 0 )); then
  echo "== SMOKE FAILED: ${FAIL}/${#SAMPLES[@]} samples did not report expected reentrancy =="
  echo "== Do NOT proceed to full-dataset run until investigated =="
  exit 1
fi
echo "== SMOKE PASSED: ${#SAMPLES[@]}/${#SAMPLES[@]} samples flagged =="
