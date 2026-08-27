# Initial Pilot Results

Date: 2026-08-11

## Dataset

- 435 manually reviewed 3-second windows
- 75 source videos
- 10 Hz sampling, producing 30 time steps per window
- 13 observable facial geometry and head-pose features
- Binary target: level 0 = `no_gaze_shift`; levels 1/2 = `gaze_shift`
- Class counts: 233 negative and 202 positive

All 435 windows were extracted successfully. Mean sampled-frame face visibility
was 0.999.

## Seed 42 held-out result

- Train: 275 windows / 44 videos
- Validation: 71 windows / 15 videos
- Test: 89 windows / 16 videos
- Test macro F1: 0.645
- Test balanced accuracy: 0.646
- Best epoch: 13

Confusion matrix (rows are actual, columns are predicted):

```text
[[35, 12],
 [19, 23]]
```

On the identical seed-42 split, summary-feature baselines produced:

| Model | Macro F1 | Balanced accuracy |
|---|---:|---:|
| Logistic Regression | 0.620 | 0.622 |
| Random Forest | 0.747 | 0.747 |
| Temporal CNN | 0.645 | 0.646 |

The Random Forest was stronger than the temporal CNN in this first experiment.
This means the new sequence extraction is useful, but the current amount of data
does not yet justify the more complex temporal model.

## Split sensitivity

Five grouped runs used seeds 1, 7, 21, 42, and 84. Each seed changed both the
held-out videos and model initialisation.

| Seed | Macro F1 | Balanced accuracy |
|---:|---:|---:|
| 1 | 0.369 | 0.392 |
| 7 | 0.711 | 0.711 |
| 21 | 0.616 | 0.616 |
| 42 | 0.645 | 0.646 |
| 84 | 0.588 | 0.593 |
| Mean | 0.586 | 0.592 |
| Standard deviation | 0.116 | 0.107 |

Across the same five grouped seeds, the Random Forest achieved mean macro F1
0.686 with standard deviation 0.047, making it both stronger and more stable than
the temporal CNN in this pilot.

## Interpretation

The frame-sequence pipeline can learn some binary separation, but performance is
not stable across held-out video groups. The seed-42 score should not be presented
alone as the expected accuracy. The five-run mean and spread are more honest for
this pilot.

This experiment does not validate brow, eye-widening, mouth, jaw, or hand-to-face
events. Those targets do not yet have enough human labels. It also does not prove
that gaze shift indicates stress, honesty, confidence, or subject knowledge.

Recommended next experiment: freeze a person-level test set, add a simple
non-temporal baseline on the same split, and create manually reviewed positive and
negative windows for exactly one new observable event.
