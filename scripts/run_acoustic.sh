#!/usr/bin/env bash
set -euo pipefail
: "${MUSIC_DATA_ROOT:?Set dataset root}"
: "${BASELINE_CHECKPOINT:?Set v2 checkpoint}"
: "${ACOUSTIC_RUN:?Set new output run directory}"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
mkdir -p "$ACOUSTIC_RUN"
trap 'echo failed > "$ACOUSTIC_RUN/stage.txt"' ERR
echo preparing > "$ACOUSTIC_RUN/stage.txt"
python -m piano_refiner.prepare_acoustic --root "$MUSIC_DATA_ROOT" --baseline "$BASELINE_CHECKPOINT"
echo training > "$ACOUSTIC_RUN/stage.txt"
python -m piano_refiner.train_acoustic --root "$MUSIC_DATA_ROOT" --baseline "$BASELINE_CHECKPOINT" --run "$ACOUSTIC_RUN" --device "${DEVICE:-cpu}" --epochs "${EPOCHS:-12}"
echo complete > "$ACOUSTIC_RUN/stage.txt"
