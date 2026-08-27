# Vision Events V2 Pilot Results

Current rerun date: 2026-08-27

## Dataset

- 435 manually reviewed 3-second windows
- 75 source videos
- 10 Hz sampling, producing 30 time steps per window
- 13 observable facial geometry and head-pose features
- Binary target: level 0 = `no_gaze_shift`; levels 1/2 = `gaze_shift`
- Class counts: 233 negative and 202 positive

All 435 windows were extracted successfully. Mean sampled-frame face visibility
was 0.999.

## Pitch-normalisation correction

The initial 2026-08-11 extraction stored the raw `solvePnP` pitch Euler angle.
A front-facing head can be represented near either 0 or +/-180 degrees, so raw
values introduced artificial wrap-around jumps into sequence statistics.

The rerun maps pitch to the signed offset from the nearest front-facing
representation before saving each feature vector. Across 13,042 finite frame
measurements, the pitch range changed from approximately `[-180, 180]` with a
standard deviation of `174.35` degrees to `[-79.77, 83.95]` with a standard
deviation of `8.14` degrees. No corrected value exceeds +/-90 degrees.

All results below were regenerated from the corrected sequence dataset. The
2026-08-11 metrics are superseded and should not be cited as current results.

## Seed 42 held-out result

- Train: 275 windows / 44 videos
- Validation: 71 windows / 15 videos
- Test: 89 windows / 16 videos

| Model | Macro F1 | Balanced accuracy |
|---|---:|---:|
| Logistic Regression | 0.643 | 0.645 |
| Random Forest | 0.749 | 0.748 |
| Temporal CNN | 0.639 | 0.650 |

The seed-42 confusion matrices (rows are actual, columns are predicted) were:

```text
Logistic Regression  [[36, 11], [20, 22]]
Random Forest        [[39,  8], [14, 28]]
Temporal CNN         [[41,  6], [24, 18]]
```

The Temporal CNN selected epoch 18 using validation macro F1. Random Forest
remained the strongest model on this split.

## Split sensitivity

Five grouped runs used seeds 1, 7, 21, 42, and 84. Each seed changed the held-out
videos; the Temporal CNN seed also changed model initialisation.

| Seed | Logistic F1 | Random Forest F1 | Temporal CNN F1 |
|---:|---:|---:|---:|
| 1 | 0.667 | 0.689 | 0.402 |
| 7 | 0.673 | 0.695 | 0.513 |
| 21 | 0.605 | 0.640 | 0.712 |
| 42 | 0.643 | 0.749 | 0.639 |
| 84 | 0.711 | 0.689 | 0.698 |
| Mean | 0.660 | 0.693 | 0.593 |
| Standard deviation | 0.035 | 0.034 | 0.118 |

| Model | Mean balanced accuracy | Standard deviation |
|---|---:|---:|
| Logistic Regression | 0.662 | 0.036 |
| Random Forest | 0.695 | 0.032 |
| Temporal CNN | 0.605 | 0.110 |

## Interpretation

Correcting pitch wrap removed a known preprocessing error without materially
changing the overall ranking: the summary-feature Random Forest remains stronger
and more stable than the Temporal CNN. The seed-42 score should not be presented
alone as expected performance; the grouped mean and spread are more honest for
this pilot.

This experiment does not validate brow, eye-widening, mouth, jaw, or hand-to-face
events. It also does not prove that gaze shift indicates stress, honesty,
confidence, or subject knowledge.

Recommended next experiment: freeze a person-level test set, audit errors by
source video, and compare raw gaze/head features with person-centred deltas on the
same grouped splits.
