# Vision Events V2 Pilot

This isolated experiment tests a new training path without changing the existing
application. It reads frames directly from `sample_vid/`, converts each labelled
3-second window into a 10 Hz facial-feature sequence, and trains a small temporal
CNN with video-level train/validation/test separation.

## What this pilot can train now

The repository already contains manually reviewed `look_away_level` labels. The
pilot maps level 0 to `no_gaze_shift` and levels 1/2 to `gaze_shift`, so it can run
without new annotation.

This is only a pipeline proof-of-concept. A new human annotation pass is required
before training brow, eye-widening, mouth-tension, or hand-to-face events. Automatic
landmark/rule outputs may suggest clips to review, but must not be used as ground
truth for the final evaluation.

## Inputs and generated artifacts

- Videos: `sample_vid/*.mp4` and `sample_vid/*.mov`
- Existing pilot labels: `data/labels/manual_window_labels.csv`
- Sequence dataset: `experiments/vision_events_v2/artifacts/look_away_sequences.npz`
- Model: `experiments/vision_events_v2/artifacts/temporal_cnn.pt`
- Metrics: `experiments/vision_events_v2/artifacts/metrics.json`

Generated artifacts are ignored by Git.

The first completed run and split-sensitivity results are recorded in
[`PILOT_RESULTS.md`](PILOT_RESULTS.md).

## Run the pilot

From the project root:

```bash
source backend/venv311/bin/activate
python experiments/vision_events_v2/extract_sequences.py
python experiments/vision_events_v2/train_baselines.py
python experiments/vision_events_v2/train_temporal_cnn.py
```

On macOS, MediaPipe may need to run from a normal Terminal session because its
FaceMesh graph creates a local OpenGL context.

For a quick smoke test:

```bash
python experiments/vision_events_v2/extract_sequences.py --max-windows 80
python experiments/vision_events_v2/train_temporal_cnn.py --epochs 30 --patience 8
```

Use `--target level` during extraction to keep the original three-class target:

```bash
python experiments/vision_events_v2/extract_sequences.py --target level
```

The binary target is the recommended first diagnostic because the current
automatic features and labels do not reliably separate look-away levels 1 and 2.

## Frame-level features

Each sampled frame produces the following observable measurements:

- face visibility;
- left/right eye aspect ratio;
- left/right eyebrow-to-eye distance;
- mouth opening and width;
- mouth-corner asymmetry;
- iris horizontal and vertical position;
- head pitch, yaw, and roll.

Head pitch is stored as a signed offset from the nearest front-facing Euler
representation. This prevents the equivalent +180/-180 representations produced
by `solvePnP` from becoming artificial jumps in sequence statistics.

Missing frames are retained and imputed from the training split only. The test
videos remain untouched until final evaluation.

## Labels for new events

`annotations/reviewed_windows.csv` records which windows a human actually reviewed.
`annotations/events.csv` records observable events inside those reviewed windows.
The two-file design distinguishes a reviewed negative window from an unreviewed
window.

Recommended first event vocabulary:

- `blink`
- `eye_widen`
- `brow_change`
- `gaze_shift`
- `hand_to_face`

Do not convert these observations into claims such as stress, dishonesty,
confidence, or competence. Those interpretations are not ground-truth facial
events and would require a separate, carefully validated research design.
