#!/usr/bin/env bash
set -euo pipefail
: "${REAL_DATA_ROOT:?Set new real dataset directory}"
: "${PIANO_CHECKPOINT:?Set Kong checkpoint}"
: "${BASELINE_CHECKPOINT:?Set v2 checkpoint}"
: "${ACOUSTIC_CHECKPOINT:?Set v3 checkpoint}"
: "${STRING_FONT:?Set licensed SoundFont}"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
mkdir -p "$REAL_DATA_ROOT"
trap 'echo failed > "$REAL_DATA_ROOT/stage.txt"' ERR
echo downloading > "$REAL_DATA_ROOT/stage.txt"
python -m piano_refiner.real_data --root "$REAL_DATA_ROOT" --font "$STRING_FONT" --evaluate-as-ready --device "${DEVICE:-cpu}"
echo extracting > "$REAL_DATA_ROOT/stage.txt"
python -m piano_refiner.extract --root "$REAL_DATA_ROOT" --checkpoint "$PIANO_CHECKPOINT" --device "${DEVICE:-cpu}"
echo preparing_acoustic > "$REAL_DATA_ROOT/stage.txt"
python -m piano_refiner.prepare_acoustic --root "$REAL_DATA_ROOT" --baseline "$BASELINE_CHECKPOINT"
echo evaluating > "$REAL_DATA_ROOT/stage.txt"
python -m piano_refiner.evaluate_real --root "$REAL_DATA_ROOT" --baseline "$BASELINE_CHECKPOINT" --checkpoint "$ACOUSTIC_CHECKPOINT" --output "$REAL_DATA_ROOT/evaluation" --device "${DEVICE:-cpu}"
echo complete > "$REAL_DATA_ROOT/stage.txt"
