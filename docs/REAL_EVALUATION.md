# Real-recording diagnostic protocol

Version 15 (frozen v2 correction model) remains the adopted listening reference.
V3 was rejected in listening despite a tiny synthetic-test F1 increase. This
experiment adds data and measurement, not another automatic model replacement.

## Source and sampling

MAESTRO v3.0.0, official source:
https://magenta.withgoogle.com/datasets/maestro

Audio is recorded acoustic piano with aligned MIDI, licensed CC BY-NC-SA 4.0.
Selected ZIP entries are fetched from the official Google storage archive over
validated HTTP ranges; the approximately 101 GB archive is not downloaded in full.
ZIP entry CRCs are checked and selected audio hashes saved; the full archive's
SHA-256 is not claimed to be verified. Download budgets prevent accidental bulk
retrieval. Third-party recordings and derivatives stay on server storage, not GitHub.

Network reads use four concurrent 1 MiB ranges with bounded retries. Test pieces
are acquired first, followed by validation and training pieces, without changing
selection or split membership. The runner extracts and evaluates after each newly
completed evaluation piece; `evaluation-progress` contains provisional results
on the available subset, not the final benchmark. Do not tune models on these
partial test results. Final results are written to `evaluation` after all stages.

The initial pilot selects 8 train, 4 validation and 8 test compositions, respecting
official split labels and deduplicating composer/title across the selected set.
Selection is a fixed hash order among pieces lasting 80–420 seconds, independent
of model outputs. It therefore is a limited, duration-filtered pilot, not a random
sample of the entire corpus. Three twelve-second crops per performance at 20%,
50% and 80% of duration yield 60 crops. Each gets three paired conditions: clean,
synthetic-string interference at +10 dB piano-to-string RMS ratio, and 0 dB.
Total: 180 clips / 36 minutes, with only 12 minutes of distinct source excerpts.

Added strings are synthesized, not recorded ensemble stems. Interference notes
are partly derived from local piano pitches to create difficult overlapping
harmonics. The known piano labels remain unchanged. This does not demonstrate
generalization to actual orchestral mixes. Preserve the source dataset's terms.

## Fixed comparison

Compare the all-candidate pool, v2 and v3 on identical proposals. Keep checkpoint
and threshold values fixed from synthetic training (v2 .46, v3 .51). No threshold
tuning, retraining or model promotion is part of this run. The existing frozen
AMT backbone was trained on MAESTRO-family data, so these clips cannot be claimed
unseen by every component; only the correction models were trained without them.

Primary matching is exact MIDI pitch with <=50 ms onset difference, one-to-one,
maximum cardinality. Count all reference notes in the 0.3–11.7 s interior, including
those absent from the candidate pool. This tolerance differs from the previous
70 ms synthetic training/report protocol; do not compare absolute scores directly.

Report overall TP/FP/FN, precision/recall/F1 and subgroup **recall with reference
counts**: bass <=MIDI48, weak velocity<=45, original key duration<=150ms, same-pitch
repeat within600ms, and dense neighboring onsets 15–120ms apart. The last two
timing features are not manual score/voice annotations. Short/dense notes are
ornament proxies, not verified grace-note labels. Groups overlap and must not be
summed. Empty group recalls are null, not zero. Also count unmatched predictions
near a reference onset with neighboring pitches or harmonic-like intervals;
these are diagnostic associations, not proofs of interference origin.

Paired confidence intervals resample whole compositions (1000 bootstrap draws)
to preserve correlations between crops. Small test size limits conclusions.
The resulting report must inform the next training design; no automatic replacement
of the preferred model is authorized by a small aggregate F1 gain alone.

```bash
export REAL_DATA_ROOT=/path/to/music-data/real-maestro-pilot
export PIANO_CHECKPOINT=/path/to/music-data/models/piano_note_pedal.pth
export BASELINE_CHECKPOINT=/path/to/music-data/dataset-v2/runs/dual-view-v2/best.pt
export ACOUSTIC_CHECKPOINT=/path/to/music-data/dataset-v2/runs/acoustic-v3/best.pt
export STRING_FONT=/path/to/music-data/fonts/GeneralUser-GS.sf2
export HF_HOME=/path/to/music-data/models/huggingface
export HF_HUB_OFFLINE=1
export DEVICE=cuda:0
bash scripts/run_real_benchmark.sh
```

Use a new output root when changing selection or conditions. `provenance.json`,
`manifest.jsonl`, `stage.txt`, evaluation `report.json` and `per_clip.json` record
data lineage, progress and results. Preparation resumes completed performances.
