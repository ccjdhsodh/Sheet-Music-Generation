# Initial v2 experiment

The initial full run uses 256 independent compositions and two acoustic variants
per composition: 512 x 12 seconds = 1.71 hours. Train/validation/test partitions
contain 384/64/64 clips, with no composition shared across partitions. Two font
families (TimGM6mb and GeneralUser GS) are rotated across acoustic variants.
Multiple programs from one font are not counted as independent font families.

The network has 581,797 trainable parameters, compared with 34,074 in the previous
small GRU correction network. More parameters alone are not evidence of improvement.

Before that run, a 16-composition, one-font pilot validates the complete pipeline.
The pilot is a functional check, not representative accuracy evidence. Pilot and
full runs must use separate directories; the pilot's scores cannot be presented
as the full experiment's scores.

All jobs run on external server storage. CPU is the initial default while existing
GPU workloads are active. Job state is stored in `job-status.json`; stage logs,
feature caches, `last.pt`, `best.pt` and `report.json` remain outside the repository.
To reproduce a GPU run, select a suitable device explicitly and retain the same
data manifest. Do not terminate other users' jobs to make a GPU available.

Five core tests cover true repeat labels, maximum-cardinality matching, padding
invariance with finite gradients, split leakage detection and chord ordering.
Tests establish implementation properties, not transcription accuracy.

The full run is not a real-audio benchmark. Before adopting v2 as a replacement,
compare v1 and v2 on the same held-out candidate pool, add an unseen-font test,
and evaluate separately on legally obtained real piano performances. Evaluate
grace/repeated notes and low-register recall as well as overall F1. Do not tune
the model or thresholds on the user's previously corrected excerpt.
