# Old 50/5 Split: Unified Evaluation of Trajectory Policies and SC-NBV v9–v24

## 1. Evaluation Protocol

- Data split: historical old 50/5 split; real unseen class IDs are `[0, 2, 26, 34, 50]`.
- Test set: 414 valid unseen episodes.
- Training seeds: `20260909, 20260910, 20260911`.
- Evaluation seeds: strictly the same set of `20260920–20260949` (30 seeds).
- Fixed starts: starting from `view0, view1, view2, view3` respectively; fixed-start results in the tables are averages over the four.
- Random start: each episode randomly selects a start among valid views; Policy and Random use the same start and path seeds.
- `moves=1`: fusing 2 views; `moves=2`: fusing 3 views.
- `Single`: starting view only; `Random`: subsequent views selected randomly; `Policy`: subsequent views selected by the model.
- The Fixed Random column is the average over the 30 evaluation seeds; intervals in the random-start table come directly from the 30 evaluation seeds.

The fixed-start four-view-average Single is **49.46%**; the random-start Single average is **49.86%**. These two baselines are independent of the model version.

The 95% CIs are two-sided Student-t intervals: for fixed starts the sample is the 3 training seeds (df=2), and for random starts the sample is the 30 evaluation seeds (df=29); `Δ` is always the paired `Policy - Random` under the same seed.

This report currently covers the completed `v1, v4, v5, v6, object_only, geometry_only`, plus the earlier `SC-NBV v9–v24`. Extraction of the v3 old-split bbox-size trajectory is still in progress; unfinished results are not included in the tables below and will be appended once completed.

SC-NBV v9–v24 use the canonical `selected_variant` determined by the validation set in each variant's configuration, together with existing checkpoints; this evaluation only re-evaluates on the unified old-split set of 414 unseen episodes and does not re-select models based on the old test set.

## 2. Fixed Start: view0–view3 Average

Percentage intervals are 95% CIs; fixed-start Random is the 30-seed average, and its interval is not expanded separately.

| Model | 2-view Policy [95% CI] | Random | Δ Policy−Random [95% CI] | 3-view Policy [95% CI] | Random | Δ Policy−Random [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| v1 | 52.36 [51.76,52.96]% | 52.06% | +0.30 [-0.30,+0.90]pp | 53.36 [52.67,54.05]% | 53.08% | +0.28 [-0.41,+0.97]pp |
| v4 | 51.85 [51.77,51.94]% | 52.06% | -0.21 [-0.29,-0.12]pp | **53.60 [53.52,53.69]%** | 53.08% | **+0.52 [+0.43,+0.61]pp** |
| v5 | 51.67 [51.44,51.90]% | 52.06% | -0.39 [-0.62,-0.16]pp | 53.22 [52.69,53.75]% | 53.08% | +0.14 [-0.39,+0.67]pp |
| v6 | 52.13 [51.18,53.09]% | 52.06% | +0.08 [-0.88,+1.03]pp | 53.50 [53.11,53.90]% | 53.08% | +0.42 [+0.02,+0.82]pp |
| object_only | 52.29 [50.92,53.67]% | 52.06% | +0.24 [-1.14,+1.61]pp | 53.12 [51.46,54.78]% | 53.08% | +0.04 [-1.62,+1.70]pp |
| geometry_only | 51.35 [50.72,51.97]% | 52.06% | **-0.71 [-1.33,-0.08]pp** | 52.80 [52.42,53.18]% | 53.08% | -0.28 [-0.66,+0.09]pp |
| v9 | 52.42 [51.22,53.61]% | 52.06% | +0.36 [-0.83,+1.55]pp | 53.40 [52.34,54.47]% | 53.08% | +0.32 [-0.74,+1.38]pp |
| v10 | 51.63 [50.80,52.47]% | 52.06% | -0.43 [-1.26,+0.41]pp | 52.36 [50.51,54.20]% | 53.08% | -0.73 [-2.57,+1.12]pp |
| v11 | 51.63 [50.98,52.28]% | 52.06% | -0.43 [-1.08,+0.23]pp | 52.07 [51.05,53.09]% | 53.08% | -1.01 [-2.03,+0.01]pp |
| v12 | 52.15 [51.48,52.83]% | 52.06% | +0.10 [-0.58,+0.77]pp | 52.48 [51.56,53.39]% | 53.08% | -0.61 [-1.52,+0.31]pp |
| v13 | 52.07 [51.32,52.83]% | 52.06% | +0.02 [-0.74,+0.77]pp | 52.96 [51.88,54.04]% | 53.08% | -0.12 [-1.20,+0.96]pp |
| v14 | 51.97 [51.22,52.73]% | 52.06% | -0.08 [-0.84,+0.67]pp | 52.96 [51.39,54.53]% | 53.08% | -0.12 [-1.69,+1.44]pp |
| v15 | 52.13 [49.37,54.90]% | 52.06% | +0.08 [-2.69,+2.84]pp | 52.70 [52.52,52.87]% | 53.08% | -0.38 [-0.56,-0.21]pp |
| v16 | 52.21 [51.22,53.21]% | 52.06% | +0.16 [-0.84,+1.16]pp | 53.02 [51.97,54.07]% | 53.08% | -0.06 [-1.11,+0.99]pp |
| v17 | 52.25 [51.50,53.01]% | 52.06% | +0.20 [-0.56,+0.95]pp | 53.30 [52.68,53.93]% | 53.08% | +0.22 [-0.41,+0.84]pp |
| v18 | 52.05 [51.22,52.89]% | 52.06% | -0.00 [-0.84,+0.83]pp | 53.44 [52.53,54.35]% | 53.08% | +0.36 [-0.55,+1.27]pp |
| v19 | 51.99 [50.71,53.27]% | 52.06% | -0.06 [-1.35,+1.22]pp | 53.48 [52.60,54.36]% | 53.08% | +0.40 [-0.48,+1.28]pp |
| v20 | 52.17 [50.65,53.70]% | 52.06% | +0.12 [-1.41,+1.64]pp | 52.92 [50.35,55.48]% | 53.08% | -0.16 [-2.73,+2.40]pp |
| v21 | 51.63 [50.50,52.76]% | 52.06% | -0.43 [-1.56,+0.71]pp | 52.68 [50.80,54.55]% | 53.08% | -0.40 [-2.28,+1.47]pp |
| v22 | 51.37 [49.76,52.97]% | 52.06% | -0.69 [-2.29,+0.92]pp | 52.88 [51.88,53.88]% | 53.08% | -0.20 [-1.20,+0.80]pp |
| v23 | 52.11 [50.59,53.64]% | 52.06% | +0.06 [-1.47,+1.58]pp | 53.32 [53.06,53.58]% | 53.08% | +0.24 [-0.02,+0.50]pp |
| v24 | 52.42 [51.82,53.02]% | 52.06% | +0.36 [-0.24,+0.96]pp | 53.44 [51.17,55.71]% | 53.08% | +0.36 [-1.91,+2.63]pp |

## 3. Random Start: 30 Evaluation Seeds

| Model | 2-view Policy [95% CI] | Random [95% CI] | Δ [95% CI] | 3-view Policy [95% CI] | Random [95% CI] | Δ [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| v1 | 52.34 [52.04,52.65]% | 51.88 [51.49,52.26]% | **+0.46 [+0.13,+0.80]pp** | 53.43 [53.26,53.60]% | 53.09 [52.85,53.34]% | **+0.34 [+0.09,+0.59]pp** |
| v4 | 51.91 [51.54,52.28]% | 51.88 [51.49,52.26]% | +0.03 [-0.43,+0.49]pp | **53.63 [53.33,53.92]%** | 53.09 [52.85,53.34]% | **+0.53 [+0.19,+0.88]pp** |
| v5 | 51.60 [51.31,51.89]% | 51.88 [51.49,52.26]% | -0.28 [-0.63,+0.08]pp | 53.32 [53.15,53.50]% | 53.09 [52.85,53.34]% | +0.23 [-0.05,+0.51]pp |
| v6 | 52.05 [51.76,52.34]% | 51.88 [51.49,52.26]% | +0.17 [-0.24,+0.59]pp | 53.48 [53.32,53.64]% | 53.09 [52.85,53.34]% | **+0.39 [+0.10,+0.67]pp** |
| object_only | 52.31 [52.01,52.62]% | 51.88 [51.49,52.26]% | **+0.44 [+0.08,+0.80]pp** | 53.14 [52.95,53.33]% | 53.09 [52.85,53.34]% | +0.05 [-0.25,+0.34]pp |
| geometry_only | 51.42 [51.13,51.72]% | 51.88 [51.49,52.26]% | **-0.45 [-0.88,-0.03]pp** | 52.97 [52.78,53.16]% | 53.09 [52.85,53.34]% | -0.13 [-0.47,+0.21]pp |
| v9 | **52.58 [52.27,52.90]%** | 51.88 [51.49,52.26]% | **+0.71 [+0.37,+1.05]pp** | 53.48 [53.31,53.65]% | 53.09 [52.85,53.34]% | **+0.38 [+0.11,+0.66]pp** |
| v10 | 51.69 [51.39,51.98]% | 51.88 [51.49,52.26]% | -0.19 [-0.53,+0.14]pp | 52.30 [52.11,52.49]% | 53.09 [52.85,53.34]% | -0.79 [-1.11,-0.48]pp |
| v11 | 51.56 [51.27,51.85]% | 51.88 [51.49,52.26]% | -0.31 [-0.69,+0.06]pp | 52.12 [51.94,52.29]% | 53.09 [52.85,53.34]% | -0.97 [-1.26,-0.69]pp |
| v12 | 52.16 [51.87,52.44]% | 51.88 [51.49,52.26]% | +0.28 [-0.12,+0.67]pp | 52.50 [52.31,52.68]% | 53.09 [52.85,53.34]% | -0.60 [-0.89,-0.30]pp |
| v13 | 52.05 [51.73,52.37]% | 51.88 [51.49,52.26]% | +0.18 [-0.20,+0.56]pp | 53.01 [52.83,53.19]% | 53.09 [52.85,53.34]% | -0.08 [-0.37,+0.21]pp |
| v14 | 51.88 [51.61,52.16]% | 51.88 [51.49,52.26]% | +0.01 [-0.37,+0.38]pp | 52.89 [52.71,53.07]% | 53.09 [52.85,53.34]% | -0.20 [-0.50,+0.10]pp |
| v15 | 52.08 [51.78,52.38]% | 51.88 [51.49,52.26]% | +0.21 [-0.18,+0.59]pp | 52.60 [52.41,52.79]% | 53.09 [52.85,53.34]% | **-0.49 [-0.79,-0.19]pp** |
| v16 | 52.28 [51.94,52.62]% | 51.88 [51.49,52.26]% | **+0.41 [+0.06,+0.75]pp** | 53.11 [52.97,53.26]% | 53.09 [52.85,53.34]% | +0.02 [-0.25,+0.30]pp |
| v17 | 52.27 [52.02,52.51]% | 51.88 [51.49,52.26]% | **+0.39 [+0.02,+0.76]pp** | 53.40 [53.23,53.56]% | 53.09 [52.85,53.34]% | **+0.30 [+0.01,+0.60]pp** |
| v18 | 51.94 [51.61,52.26]% | 51.88 [51.49,52.26]% | +0.06 [-0.36,+0.48]pp | 53.43 [53.29,53.58]% | 53.09 [52.85,53.34]% | **+0.34 [+0.08,+0.60]pp** |
| v19 | 52.04 [51.69,52.38]% | 51.88 [51.49,52.26]% | +0.16 [-0.25,+0.57]pp | **53.51 [53.36,53.65]%** | 53.09 [52.85,53.34]% | **+0.41 [+0.16,+0.67]pp** |
| v20 | 52.11 [51.85,52.38]% | 51.88 [51.49,52.26]% | +0.24 [-0.14,+0.61]pp | 53.02 [52.82,53.21]% | 53.09 [52.85,53.34]% | -0.08 [-0.34,+0.19]pp |
| v21 | 51.75 [51.45,52.06]% | 51.88 [51.49,52.26]% | -0.12 [-0.46,+0.21]pp | 52.74 [52.58,52.89]% | 53.09 [52.85,53.34]% | -0.35 [-0.65,-0.05]pp |
| v22 | 51.36 [51.04,51.67]% | 51.88 [51.49,52.26]% | **-0.52 [-0.85,-0.18]pp** | 52.88 [52.72,53.04]% | 53.09 [52.85,53.34]% | -0.21 [-0.54,+0.12]pp |
| v23 | 52.34 [52.06,52.63]% | 51.88 [51.49,52.26]% | **+0.47 [+0.16,+0.77]pp** | 53.31 [53.15,53.48]% | 53.09 [52.85,53.34]% | +0.22 [-0.06,+0.51]pp |
| v24 | 52.50 [52.18,52.82]% | 51.88 [51.49,52.26]% | **+0.62 [+0.27,+0.97]pp** | 53.47 [53.32,53.63]% | 53.09 [52.85,53.34]% | **+0.38 [+0.08,+0.68]pp** |

## 4. How to Read the Results

1. Under the old-split random-start protocol, the full trajectory policy v1 exceeds Random at both move counts: `52.34% vs 51.88%`, `53.43% vs 53.09%`, with paired 95% CIs not crossing 0.
2. Among SC-NBV v9–v24, v9 has the most prominent random-start results: 1 move at **52.58%** and 2 moves at **53.48%**; the CIs relative to Random are above 0 in both cases. v24 also exceeds Random at both move counts, by **+0.62pp** and **+0.38pp** respectively, with CIs not crossing 0.
3. v4 is among the best for 2 moves: **53.63%** on random start, Δ=`+0.53pp`; but for 1 move it gains only `+0.03pp`, so it cannot be called stably better than Random at both move counts.
4. v17, v18, v19 show positive gains at 2 moves; v16 and v23 show gains mainly at 1 move. v10, v11, v15, v21, v22 are clearly below Random in at least one protocol.
5. `geometry_only` is significantly below Random at 1 move with random starts, showing that candidate geometry alone is insufficient to predict recognition gain; temporal object/human or direct sensor summaries still matter.
6. These differences are typically only about `0.3–0.7pp`; the 30 seeds mainly characterize path randomness and do not equal 30 independent test sets, so conclusions should be stated as "a stable advantage observed on this old split under this evaluation protocol" and not generalized into a universal statistical claim.

## 5. Versions and Result Files

- Trajectory-policy old-split results: `work_dir/trajectory_ablation_old_split/eval_30seeds_recheck/`
- SC-NBV v9–v24 old-split results: `work_dir/sc_nbv_v9_v24_old_split_30seeds/`
- SC-NBV v9–v24 evaluation log: `logs/sc_nbv_v9_v24_old_split_30seeds.log`
- Trajectory-policy evaluation logs: `logs/trajectory_ablation_old_split_*_30seeds_recheck.log`
- v3 bbox-size extraction log: `logs/trajectory_ablation_old_split_v3_bbox_extract.log`

The `summary.json` in each variant directory stores the per-seed details for the 30 seeds; the CIs in this report are recomputed from these raw details.
