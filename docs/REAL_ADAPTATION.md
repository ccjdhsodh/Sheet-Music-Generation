# Real-data adaptation, controlled pilot

The real-recording diagnostic found that candidate filtering removes many valid
notes, with weak, bass and repeated notes particularly affected. This experiment
changes training data and the objective, while keeping the acoustic residual
architecture fixed. It does not add song-specific corrections or claim that the
architecture has been replaced.

Two arms start independently from the same zero residual over frozen v2 logits:

1. `data_only`: mix real and synthetic training clips, ordinary cross entropy.
2. `retention`: identical sampling and initialization, with higher positive-label
   weights for bass (<=48), weak (velocity<=45), short (<=150ms) and true repeats.
   Weights start at 1, add .5 for each of the first three and 1 for repeats, capped
   at 3. False notes retain weight 1; this is not blanket repeat preservation.

Both use 50/50 domain sampling, 256 sampled clips per epoch, 10 epochs, AdamW
1e-4, and the same residual regularization. Only the 8 real training compositions
(72 variants/crops) and the pre-existing synthetic training split train the model.
The 4 real validation compositions select thresholds/checkpoints. The trainer
does not load test examples. Training labels retain the existing 70ms matching;
validation rematches selected notes at 50ms within each crop's interior.

Selection maximizes mean F1 across the three validation interference conditions,
subject to precision being within 1 percentage point of v2 and all/bass/weak/
short/repeated recall each being within .5 points of v2, separately per condition.
An unchanged v2 fallback is saved before training. These are validation safeguards,
not guarantees for unseen music. The already-inspected test split is a regression
set; final generalization claims require an additional untouched test set.

```bash
export REAL_DATA_ROOT=/data1/chencong/music-data/maestro-real-pilot-v1
export SYNTHETIC_ROOT=/data1/chencong/music-data/dataset-v2
export BASELINE_CHECKPOINT=$SYNTHETIC_ROOT/runs/dual-view-v2/best.pt
export ADAPTATION_RUN=/data1/chencong/music-data/runs/real-adaptation-v4
export DEVICE=cuda:0
bash scripts/run_real_adaptation.sh
```

Inspect each arm's `protocol.json`, `history.jsonl`, `report.json` and `best.pt`.
No automatic listening-version promotion occurs. Candidate recovery outside the
existing proposal pool remains a separate, unresolved problem.
