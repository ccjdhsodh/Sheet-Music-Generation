#!/usr/bin/env bash
set -euo pipefail
: "${REAL_DATA_ROOT:?}" "${SYNTHETIC_ROOT:?}" "${BASELINE_CHECKPOINT:?}" "${ATTACK_RUN:?}"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
mkdir -p "$ATTACK_RUN"
trap 'echo failed > "$ATTACK_RUN/stage.txt"' ERR
for arm in retention discrimination; do
  echo "$arm" > "$ATTACK_RUN/stage.txt"
  python -m piano_refiner.train_real --real-root "$REAL_DATA_ROOT" --synthetic-root "$SYNTHETIC_ROOT" --baseline "$BASELINE_CHECKPOINT" --run "$ATTACK_RUN/$arm" --arm "$arm" --architecture attack --precision-slack 0 --device "${DEVICE:-cpu}"
done
echo complete > "$ATTACK_RUN/stage.txt"
