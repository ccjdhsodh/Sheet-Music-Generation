#!/usr/bin/env bash
set -euo pipefail
: "${REAL_DATA_ROOT:?}" "${SYNTHETIC_ROOT:?}" "${BASELINE_CHECKPOINT:?}" "${ADAPTATION_RUN:?}"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
mkdir -p "$ADAPTATION_RUN"
trap 'echo failed > "$ADAPTATION_RUN/stage.txt"' ERR
for arm in data_only retention; do
  echo "$arm" > "$ADAPTATION_RUN/stage.txt"
  python -m piano_refiner.train_real --real-root "$REAL_DATA_ROOT" --synthetic-root "$SYNTHETIC_ROOT" --baseline "$BASELINE_CHECKPOINT" --run "$ADAPTATION_RUN/$arm" --arm "$arm" --device "${DEVICE:-cpu}"
done
echo complete > "$ADAPTATION_RUN/stage.txt"
