# Acoustic residual experiment (v3)

V2 uses only frozen AMT posteriors and event context. V3 adds evidence from the
original mixture waveform: 1024/4096-point STFT pitch-band maps and short-window
spectral change, at 100 Hz. Each candidate gets a 3 x 49 x 41 patch covering
-12 to +36 semitones and -160 to +240 ms around its onset. Clip-level gain
normalization avoids amplifying each weak candidate independently.

A small 2-D CNN encodes the audio. A bidirectional GRU combines that encoding
with frozen v2 logits, context and position. A zero-initialized residual head
starts exactly at v2 predictions and learns changes to action logits. V2 stays
frozen. This is not a source-separation retraining or supervised voice assignment;
voice-like relationships are learned only through event context.

The first experiment reuses the existing 512 synthetic clips and their original
composition splits, widening the evidence rather than adding song-specific labels.
No listening-feedback timestamps enter training, inference or threshold selection.
The code adds no new corpus in this experiment. All three action classes remain:
reject, true onset, and true same-key reattack. Original velocities and durations
are retained during the comparison with listening version 15.

Selection uses validation F1, subject to recall and genuine-reattack candidate
retention dropping by at most 0.5 percentage points versus v2 on validation.
The unchanged baseline is a valid fallback if no checkpoint improves that score.
These safeguards constrain validation performance, not guaranteed real-recording
behavior. Test metrics are paired on the same candidate pool and evaluated only
after selection. This test set was already used for v2 reporting, so it is not
a fresh external benchmark. Synthetic data cannot establish real-audio superiority.

```bash
export MUSIC_DATA_ROOT=/path/to/music-data/dataset-v2
export BASELINE_CHECKPOINT=$MUSIC_DATA_ROOT/runs/dual-view-v2/best.pt
export ACOUSTIC_RUN=$MUSIC_DATA_ROOT/runs/acoustic-v3
export DEVICE=cuda:0
bash scripts/run_acoustic.sh
```

Feature caches are versioned and tied to the baseline checkpoint SHA-256.
Preparation resumes completed files; use a new run directory for a fresh training
attempt. `stage.txt`, `history.jsonl`, `best.pt` and `report.json` expose progress.
`infer_acoustic` uses 12-second windows and 6-second overlap for long recordings;
it requires original audio and existing mix/piano posterior caches. The window
protocol matches the earlier v2 real-recording trial.
