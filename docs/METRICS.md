# Metric settings

Every setting below is what produced the numbers in the paper. The code is the reference; each module's docstring
repeats the rule it implements.

## Shot segmentation (`msbench/shots.py`, `msbench/transitions.py`)

| Step | Setting |
|---|---|
| Hard cuts | OmniShotCut v1.5, `Hard_Cut` / `Sudden_Jump` relations |
| Gradual intervals | TransVLM-Qwen3-VL-4B (official prompt, HF backend); dropped if within 0.2 s of a hard cut |
| Interval type | OmniShotCut dissolve / fade / wipe label of the most-overlapping segment, else the judge's 4-class answer (`prompts/transition_type4.txt`, thinking low, video at 10 fps) |
| Removal | interval without OmniShotCut label dropped if the judge says *hard cut*, or the 6-class judge answer (`prompts/transition_type6.txt`) is *none* |
| `clean` shots | spans between events, gradual intervals removed, 2 frames trimmed at each hard cut (AS, PS, DD, CC) |
| `m01` shots | 0.1 s margin around each event, minimum 0.3 s (VQA windows, face) |

## Sub-metrics

| Sub-metric | Module | Input | Score | Video aggregation | Range |
|---|---|---|---|---|---|
| Aesthetic | `metrics/visual.py` | clean shots, 4 fps | Q-Align "quality of this video" prompt, 5-level softmax | duration-weighted mean | 0-1 |
| Perceptual | `metrics/visual.py` | clean shots, 8 fps | DeQA-Score-Mix3 frame mean | duration-weighted mean / 5 | 0-1 |
| Dynamic Degree | `metrics/devil.py` | clean shots ≥ 0.4 s, ≤ 8 fps | 7 DEVIL components | shot mean → z-score → ridge (`configs/devil_ridge.json`) | 1-5 |
| Physical Plausibility | `metrics/vlm.py` | whole video, 10 fps | `prompts/pp.txt`, expected score from top-5 logprobs | - | 1-5 |
| VQA | `metrics/vlm.py` | whole video, 4 fps | `prompts/vqa.txt`, P(yes) per line, soft rule; shot windows < 1 s dropped | mean over questions | 0-1 |
| Transition Control | `metrics/timing.py` | detected events | correct type / requested after order-preserving alignment | - | 0-1 |
| Cut Time Accuracy | `metrics/timing.py` | detected shots | exp(−error / duration) | - | 0-1 |
| Camera Control | `metrics/camera.py` | clean shots: frames at 2 fps, FFV1 clip | FilmOps labels vs. script, partial credit | mean of scale / angle / movement over planned shots | 0-1 |
| Entity consistency | `metrics/vlm.py` | whole video, 10 fps | `prompts/csec.txt`, expected score | - | 1-5 |
| Background consistency | `metrics/vlm.py` | whole video, 10 fps | `prompts/csbc.txt`, expected score | - | 1-5 |
| Face consistency | `metrics/face.py` | m01 shots, 4 fps | InsightFace buffalo_l + Set-of-Mark (`prompts/face_som.txt`), calibrated similarity | mean over entities | 0-1 |

Judge decoding: `off_official` (no thinking, temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5) for PP, VQA,
CSEC and CSBC; thinking with low effort (temperature 1.0, top-p 0.95, top-k 20) for transition types; greedy for
Set-of-Mark. Seed 2027 everywhere.

## Aggregation (`msbench/aggregate.py`)

1. 1-5 scores map to [0, 1] by (x − 1) / 4.
2. Cross-shot metrics apply only where the script's `eval_metrics` asks for them. An applicable video with fewer than
   two detected shots (face: or without the person in two shots) scores 0.
3. Model level, per duration split: each sub-metric is averaged over the videos it applies to; TC = TC-V × TC-A
   (TC-V = matched / requested, TC-A = correct / matched); each dimension is the equal-weight mean of its sub-metrics;
   the Final Score is the equal-weight mean of VQ, DQ, SA and CSC.
