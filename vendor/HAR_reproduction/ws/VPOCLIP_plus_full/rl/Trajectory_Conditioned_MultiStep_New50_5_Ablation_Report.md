# Trajectory-Conditioned Multi-Step Policy: Full Ablation on the New 50/5 Split

## 1. Experimental Protocol

- Data: new 50/5 split; the real unseen classes are `A014, A015, A025, A039, A046`.
- Evaluation samples: 271 valid unseen episodes.
- Each policy uses 3 training seeds: `20260909, 20260910, 20260911`.
- Random paths use 30 evaluation seeds: `20260920–20260949`.
- Fixed starts are `view0, view1, view2, view3` respectively; a protocol with a random start per sample is also included.
- `moves=1` means fusing 2 views; `moves=2` means fusing 3 views.
- `Single` uses only the starting view; `Random` selects subsequent views randomly; `Policy` selects them with the policy; `Oracle` traverses candidate views with ground-truth labels and serves only as an upper bound.

Note: no standalone trajectory-v2 code or checkpoint was found, so this evaluation actually covers `v1, v3, v4, v5, v6`, plus the two follow-up data ablations `object_only, geometry_only`; no v2 results were fabricated.

## 2. Fixed-Start Results Averaged over Four Views

Averaged over the four fixed starts `view0–view3`, the table below shows Top-1. Percentages are rounded to 2 decimal places; `Δ` is `Policy - Random`.

| Version | 2-view Single | 2-view Random | 2-view Policy | 2-view Oracle | Δ | 3-view Single | 3-view Random | 3-view Policy | 3-view Oracle | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 | 62.45% | 64.55% | 64.21% | 71.13% | -0.34pp | 62.45% | 64.11% | 63.87% | 68.54% | -0.25pp |
| v3 | 62.45% | 64.55% | 64.08% | 71.13% | -0.47pp | 62.45% | 64.11% | 63.81% | 68.54% | -0.31pp |
| v4 | 62.45% | 64.55% | **65.16%** | 71.13% | **+0.61pp** | 62.45% | 64.11% | **64.30%** | 68.54% | **+0.18pp** |
| v5 | 62.45% | 64.55% | 64.33% | 71.13% | -0.22pp | 62.45% | 64.11% | 64.08% | 68.54% | -0.03pp |
| v6 | 62.45% | 64.55% | 64.33% | 71.13% | -0.22pp | 62.45% | 64.11% | 63.71% | 68.54% | -0.40pp |
| object_only | 62.45% | 64.55% | 64.85% | 71.13% | **+0.30pp** | 62.45% | 64.11% | **64.54%** | 68.54% | **+0.43pp** |
| geometry_only | 62.45% | 64.55% | 63.87% | 71.13% | -0.68pp | 62.45% | 64.11% | 63.87% | 68.54% | -0.25pp |

## 3. Per-Start Breakdown for Fixed Starts

The following are Policy Top-1 values; columns are ordered `view0 / view1 / view2 / view3` in all cases.

| Version | 1 move, 2 views fused | 2 moves, 3 views fused |
|---|---|---|
| v1 | 64.33 / 65.56 / 63.35 / 63.59% | 64.33 / 63.22 / 64.45 / 63.47% |
| v3 | 64.33 / 63.71 / 64.21 / 64.08% | 62.98 / 64.45 / 64.08 / 63.71% |
| v4 | **65.68 / 64.58 / 64.70 / 65.68%** | 63.47 / **64.70 / 64.58** / 64.45% |
| v5 | 63.96 / 65.44 / 63.59 / 64.33% | 63.84 / 63.84 / 63.84 / 64.82% |
| v6 | 63.71 / 65.19 / 63.84 / 64.58% | 63.71 / 63.22 / 63.96 / 63.96% |
| object_only | **65.81** / 64.58 / 64.08 / 64.94% | **64.82 / 64.94** / 64.21 / 64.21% |
| geometry_only | 63.96 / 64.58 / 62.85 / 64.08% | 62.98 / 63.47 / 64.21 / 64.82% |

For fixed starts, the baselines shared by all variants are:

| Start | Single | Random (2-view) | Oracle (2-view) | Random (3-view) | Oracle (3-view) |
|---|---:|---:|---:|---:|---:|
| view0 | 61.25% | 64.88% | 70.85% | 64.19% | 68.63% |
| view1 | 64.58% | 64.53% | 72.69% | 63.84% | 68.63% |
| view2 | 63.47% | 64.38% | 71.22% | 63.90% | 68.27% |
| view3 | 60.52% | 64.42% | 69.74% | 64.53% | 68.63% |

## 4. Random Start, 30 Evaluation Seeds

`±` is the path variability caused by the 30 evaluation seeds; Policy is first averaged over the 3 training seeds, then aggregated across the 30 evaluation seeds.

| Version | 2-view Single | Random | Policy | Oracle | Δ | 3-view Single | Random | Policy | Oracle | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 | 62.51% | 64.46±1.08% | 64.32±0.98% | 70.76% | -0.14pp | 62.51% | 64.35±0.81% | 63.95±0.65% | 68.62% | -0.40pp |
| v3 | 62.51% | 64.46±1.08% | 64.07±0.84% | 70.76% | -0.39pp | 62.51% | 64.35±0.81% | 63.69±0.46% | 68.62% | -0.66pp |
| v4 | 62.51% | 64.46±1.08% | **65.25±0.89%** | 70.76% | **+0.78pp** | 62.51% | 64.35±0.81% | 64.15±0.58% | 68.62% | -0.20pp |
| v5 | 62.51% | 64.46±1.08% | 64.34±0.75% | 70.76% | -0.12pp | 62.51% | 64.35±0.81% | 64.12±0.50% | 68.62% | -0.23pp |
| v6 | 62.51% | 64.46±1.08% | 64.46±0.78% | 70.76% | -0.01pp | 62.51% | 64.35±0.81% | 63.96±0.52% | 68.62% | -0.40pp |
| object_only | 62.51% | 64.46±1.08% | **64.72±0.97%** | 70.76% | **+0.25pp** | 62.51% | 64.35±0.81% | **64.65±0.54%** | 68.62% | **+0.30pp** |
| geometry_only | 62.51% | 64.46±1.08% | 63.96±0.96% | 70.76% | -0.51pp | 62.51% | 64.35±0.81% | 64.01±0.56% | 68.62% | -0.34pp |

## 5. 95% Confidence Intervals

For the mean Top-1, the CI here is the two-sided Student-t 95% CI, not a binomial interval for individual episodes:

- Random start: the 30 evaluation seeds are the sample; `Δ` is the paired difference `Policy - Random` under the same seed.
- Fixed-start four-view average: the 3 training seeds are the sample, so the intervals are wider; they reflect variation across model training seeds.
- The fixed-start Random mean is identical across the three training seeds; its 30-seed variability is already given as `±` in Section 4.

### 5.1 Random Start: 30-Seed 95% CI

Format is `mean [95% CI lower, upper]`; interval units within percentages are percentage points.

| Version | 2-view Policy | 2-view Random | 2-view Δ | 3-view Policy | 3-view Random | 3-view Δ |
|---|---:|---:|---:|---:|---:|---:|
| v1 | 64.32 [63.95,64.69]% | 64.46 [64.05,64.88]% | -0.14 [-0.62,0.34]pp | 63.95 [63.71,64.20]% | 64.35 [64.05,64.66]% | -0.40 [-0.73,-0.08]pp |
| v3 | 64.07 [63.75,64.39]% | 64.46 [64.05,64.88]% | -0.39 [-0.82,0.04]pp | 63.69 [63.52,63.86]% | 64.35 [64.05,64.66]% | -0.66 [-0.99,-0.34]pp |
| v4 | **65.25 [64.91,65.58]%** | 64.46 [64.05,64.88]% | **+0.78 [+0.28,+1.29]pp** | 64.15 [63.93,64.37]% | 64.35 [64.05,64.66]% | -0.20 [-0.51,0.11]pp |
| v5 | 64.34 [64.06,64.63]% | 64.46 [64.05,64.88]% | -0.12 [-0.57,0.32]pp | 64.12 [63.93,64.31]% | 64.35 [64.05,64.66]% | -0.23 [-0.53,0.06]pp |
| v6 | 64.46 [64.16,64.75]% | 64.46 [64.05,64.88]% | -0.01 [-0.51,0.49]pp | 63.96 [63.76,64.15]% | 64.35 [64.05,64.66]% | -0.40 [-0.75,-0.04]pp |
| object_only | 64.72 [64.35,65.08]% | 64.46 [64.05,64.88]% | +0.25 [-0.27,0.77]pp | **64.65 [64.45,64.86]%** | 64.35 [64.05,64.66]% | **+0.30 [+0.01,+0.59]pp** |
| geometry_only | 63.96 [63.59,64.32]% | 64.46 [64.05,64.88]% | -0.51 [-1.04,0.02]pp | 64.01 [63.80,64.23]% | 64.35 [64.05,64.66]% | -0.34 [-0.69,0.01]pp |

### 5.2 Fixed-Start Four-View Average across 3 Training Seeds: 95% CI

| Version | 2-view Policy | 2-view Δ | 3-view Policy | 3-view Δ |
|---|---:|---:|---:|---:|
| v1 | 64.21 [63.16,65.26]% | -0.34 [-1.39,0.71]pp | 63.87 [62.00,65.73]% | -0.25 [-2.11,1.62]pp |
| v3 | 64.08 [63.13,65.04]% | -0.47 [-1.42,0.49]pp | 63.81 [62.18,65.43]% | -0.31 [-1.93,1.32]pp |
| v4 | **65.16 [64.68,65.64]%** | **+0.61 [+0.13,+1.09]pp** | 64.30 [63.51,65.09]% | +0.18 [-0.61,0.98]pp |
| v5 | 64.33 [63.25,65.41]% | -0.22 [-1.30,0.86]pp | 64.08 [62.68,65.48]% | -0.03 [-1.43,1.37]pp |
| v6 | 64.33 [62.25,66.41]% | -0.22 [-2.30,1.86]pp | 63.71 [63.24,64.19]% | -0.40 [-0.88,0.08]pp |
| object_only | 64.85 [64.39,65.31]% | +0.30 [-0.16,0.76]pp | **64.54 [64.02,65.07]%** | +0.43 [-0.10,0.96]pp |
| geometry_only | 63.87 [62.94,64.79]% | -0.68 [-1.61,0.24]pp | 63.87 [63.60,64.13]% | -0.25 [-0.51,0.02]pp |

Under this CI definition, for random start with 1 move, v4's advantage interval does not cross 0; with 2 moves it no longer holds. The `object_only` Δ interval for random start with 2 moves is only marginally above 0, a borderline result that should not be over-interpreted.

## 6. Conclusions

1. The most stable full policy on the new split is **v4 (human orientation added)**: with 1 move it beats Random by `+0.61pp` on fixed-start average and `+0.78pp` on random start; but the advantage disappears with 2 moves, indicating the multi-step state update still needs improvement.
2. Among the data ablations, **object_only** is the most promising: it exceeds fixed-start Random at both move counts, by `+0.30pp` and `+0.43pp` respectively; on random start by `+0.25pp` and `+0.30pp` respectively.
3. **geometry_only** degrades clearly, showing that candidate geometry and the current relative orientation alone are insufficient; retaining temporal object information or human orientation is valuable.
4. v1, v3, v5, v6 do not stably exceed Random overall on the new unseen split. Oracle is still about 4–7 percentage points higher, indicating a clear gap between "a valuable view exists" and "predicting that view from the current state".
5. These gains are only about `0.25–0.78pp`, and although 271 episodes are evaluated with 30 evaluation seeds, the seed repetitions mainly measure path randomness and are not equivalent to independent dataset repetitions; a universally statistically significant improvement should not be claimed at this stage.

## 7. Result Files

- Overall evaluation results directory: `/home/youhan/ws/VPOCLIP_plus_full/work_dir/trajectory_ablation_new50_5/eval_30seeds/`
- New-split evaluation script: `/home/youhan/ws/VPOCLIP_plus_full/rl/evaluate_trajectory_variants_new_split.py`
- v3/v4/object_only/geometry_only training logs: `/home/youhan/ws/VPOCLIP_plus_full/logs/trajectory_ablation_new50_5_*_train.log`
- Each variant's `summary.json` records the full per-seed details for the 30 evaluation seeds.
