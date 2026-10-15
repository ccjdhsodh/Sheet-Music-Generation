#!/usr/bin/env bash
set -euo pipefail
: "${MUSIC_DATA_ROOT:?Set MUSIC_DATA_ROOT to external data storage}"
: "${PIANO_CHECKPOINT:?Set PIANO_CHECKPOINT to trusted Kong checkpoint}"
: "${PIANO_FONTS:?Set PIANO_FONTS to colon-separated SoundFont paths}"
IFS=: read -r -a fonts <<< "$PIANO_FONTS"
python -m piano_refiner.synthesize --root "$MUSIC_DATA_ROOT" --fonts "${fonts[@]}" --compositions "${COMPOSITIONS:-256}" --variants 2
python -m piano_refiner.extract --root "$MUSIC_DATA_ROOT" --checkpoint "$PIANO_CHECKPOINT" --device "${DEVICE:-cpu}"
python -m piano_refiner.train --root "$MUSIC_DATA_ROOT" --run "$MUSIC_DATA_ROOT/runs/dual-view-v2" --device "${DEVICE:-cpu}" --epochs "${EPOCHS:-40}"
