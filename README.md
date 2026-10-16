# Sheet Music Generation

Following real-listening regression in v3, the adopted reference remains v2
(listening version 15). The next step is a [real-recording diagnostic benchmark](docs/REAL_EVALUATION.md),
with paired clean/interference conditions and subgroup recall, not another model promotion.

The next experiment adds original-waveform evidence to the frozen v2 model:
see [the v3 experiment protocol](docs/EXPERIMENT_V3.md). It is experimental;
training completion and measured results must be checked in the external run logs.

Experimental piano transcription refinement. The current milestone is **audio-to-MIDI note refinement**, not finished sheet-music engraving.

## Network v2

A shared CNN separately encodes mixture and separated-piano posterior patches. A learned gate fuses these views. Three Transformer layers model event context using continuous time/pitch features. A three-way action head predicts rejected candidates, valid onsets and valid same-key reattacks; auxiliary heads predict velocity and key-hold duration. Wider candidate proposals permit recovery, but pitches absent from every proposal cannot be recovered.

The Kong transcription backbone and Demucs separator remain frozen. This is a new correction network, **not** an end-to-end AMT retraining. No recording-specific time edits are included.

## Data v2

Independent procedural compositions contain grace notes, real weak reattacks, chords and quiet bass. Acoustic variants add sustain pedal, strings/vibrato, noise, variable reverb and multiple SoundFonts. All variants of a composition remain in one train/validation/test split. No user audio or manually corrected MIDI is used as a training label. Fonts and generated metadata are hashed/recorded outside Git.

## Setup

Python 3.10+; use a PyTorch build compatible with the GPU architecture. The server environment inherits PyTorch 2.1.2+cu118 with Pascal support. Keep environments and assets outside Git.

```bash
python -m venv --system-site-packages /path/to/music-data/venv
source /path/to/music-data/venv/bin/activate
pip install 'numpy<2' 'librosa==0.10.2.post1' soundfile pretty_midi
# Offline synthesis does not require the PyAudio playback dependency.
pip install --no-deps tinysoundfont
pip install demucs==4.1.0 piano_transcription_inference==0.0.6
pip install --no-deps -e .
python -m unittest discover -s tests -v
```

## Pipeline

```bash
export MUSIC_DATA_ROOT=/path/to/music-data/dataset-v2
export PIANO_CHECKPOINT=/path/to/music-data/models/piano_note_pedal.pth
export PIANO_FONTS=/path/to/fonts/TimGM6mb.sf2:/path/to/fonts/GeneralUser-GS.sf2
export HF_HOME=/path/to/music-data/models/huggingface
export DEVICE=cpu # select an available GPU after inspecting current workloads
export COMPOSITIONS=256 # 512 twelve-second clips = 1.71 hours
bash scripts/run_pipeline.sh
```

Generation and extraction resume completed clips; do not run concurrent writers. Use a fresh dataset directory when changing generator configuration, fonts, checkpoint or proposal parameters. Training accepts `--resume` from its last checkpoint. No implicit dataset downloads. Missing/empty proposals fail loudly instead of dropping hard examples.

## Evaluation and limitations

Checkpoint and threshold selection uses validation only; final test reports precision/recall/F1 including unproposed ground-truth notes as false negatives, genuine reattack retention, and duration/velocity errors on matched candidates. Default inference preserves candidate velocity and duration; predicted duration is opt-in via `--use-duration`.

Current data remain synthetic. The same procedural grammar and font pool are shared across splits, although compositions are distinct. Multiple fonts do not establish unseen-timbre or real-performance generalization. No v2 accuracy improvement is claimed before completed evaluation. Key-hold duration differs from acoustic tails and score values. Pedal labels are saved but there is no pedal prediction head. MAESTRO is a future real-data evaluation stage, not included in current results.

## Sources and asset licenses

- [Kong et al.](https://arxiv.org/abs/2010.01815), [official checkpoint](https://zenodo.org/records/4034264).
- [Demucs](https://github.com/adefossez/demucs).
- [Onsets and Frames](https://arxiv.org/abs/1710.11153), inspiration for distinguishing attack and sustain evidence; not a reproduction.
- [Sigtia et al.](https://arxiv.org/abs/1508.01774), acoustic/context integration; this project does not implement their RNN-RBM.
- [GeneralUser GS](https://github.com/mrbumpy409/GeneralUser-GS), S. Christian Collins; retain the author's license with downloaded assets.
- TimGM6mb bundled with pretty_midi: retain upstream licensing alongside the font.
- [MAESTRO](https://magenta.withgoogle.com/datasets/maestro), CC BY-NC-SA 4.0; not currently included.

Source code only: no audio, weights, SoundFonts, credentials or personal SSH configuration in Git. Third-party assets retain their own licenses.
