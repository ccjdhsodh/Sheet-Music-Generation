# Attack evidence and hard-negative pilot

The v4 retention objective did not resolve the user's reported extra quiet notes.
The reported times are used only for diagnosis and final inspection, never as
training labels, selection criteria, pitch bans or inference rules.

V5 adds a temporal convolution branch over the full 12-channel, 61-frame
mixture/separated-piano AMT patch. It is fused with the existing independent
audio encoder before bidirectional sequence modeling. This exposes the onset,
offset, sustain and velocity trajectories directly to the correction network,
rather than relying on the frozen v2 logits to summarize them. The residual
starts at zero, preserving v2 before training.

Two arms use the same new architecture, seed, real/synthetic training split,
sampling and validation rules. `retention` retains v4 positive-label weighting.
`discrimination` additionally weights labeled negatives by 1 + 2 * frozen-v2
positive probability + a related-note indicator. The indicator covers same pitch
and upper intervals 12,19,24 within .6 seconds after another proposal. Positive
labels, including true repeats, never receive the negative penalty. Labels are
from aligned training MIDI, not from the user's song. This is hard-negative
weighting, not proof that any specific false note is a harmonic.

Per-condition validation precision must now be at least v2's precision. All,
bass, weak, short and repeated recall may drop by at most .5 percentage point.
Threshold/checkpoint selection otherwise uses the v4 validation protocol.
The old architecture remains loadable and existing checkpoints remain unchanged.
Two new tests verify branch gradients/padding and protection of true repeats.

Run `scripts/run_attack_adaptation.sh` with REAL_DATA_ROOT, SYNTHETIC_ROOT,
BASELINE_CHECKPOINT, ATTACK_RUN and DEVICE. This small-data pilot may fail to
improve; no checkpoint is automatically promoted. It does not implement a new
chord-recognition model, recover notes outside the proposal pool, or establish
performance on unseen real ensembles.
