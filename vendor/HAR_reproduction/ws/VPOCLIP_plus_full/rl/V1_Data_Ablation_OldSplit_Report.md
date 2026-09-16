# v1 State-Data Ablation Experiment

## 1. Purpose

Determine which state-data branches contribute to the v1 improvement, and check whether removing a single branch still beats random paths on the real unseen 5-way task.

The v1 state is split into three branches:

1. **Geometry branch**: current human/camera relative orientation, candidate view directions, candidate-relative-to-current view directions, reachability;
2. **Human trajectory branch**: 13 frames of human torso center and velocity;
3. **Object trajectory branch**: 13 frames of presence, center position, and confidence for 50 object slots, plus position relative to the human center.

This experiment does not modify VPOCLIP and does not feed future view features into the policy; all policies are still trained with the offline seen-class utility target.

## 2. Protocol

- Historical old split: real unseen classes `[0, 2, 26, 34, 50]`;
- Test samples: 414 complete three-view fusion samples;
- Each new ablation variant: 3 training seeds (20260909, 20260910, 20260911);
- Fixed-start results: averaged over the four starts view0, view1, view2, view3;
- Random-start results: a random start per sample from valid view0–3, 30 evaluation seeds (20260920–20260949);
- Policy and Random use the same start and random-path seeds;
- Moving 1 time fuses 2 views; moving 2 times fuses 3 views;
- Movement cost is not included in the training reward.

All variants follow the historical training script's pseudo-unseen top-1 gain for checkpoint selection. In this batch of experiments this metric happens to take the same discrete value, so the `best.pt` of v1, v4, v5, v6, and the two new ablations are all epoch 1; this is a protocol-level phenomenon common to all variants, not early stopping unique to the new ablations.

## 3. Fixed-Start Results

The table below shows top-1 averaged over the four fixed starts. Random is averaged over 30 evaluation seeds.

| Version | Retained state branches | 1-move Policy | Random | Diff | 2-move Policy | Random | Diff |
|---|---|---:|---:|---:|---:|---:|---:|
| geometry_only | Geometry | 51.35% | 52.06% | -0.71 | 52.80% | 53.08% | -0.28 |
| v6 / person_only | Geometry + human trajectory | 52.13% | 52.06% | +0.08 | 53.50% | 53.08% | **+0.42** |
| object_only | Geometry + object trajectory | 52.29% | 52.06% | **+0.24** | 53.12% | 53.08% | +0.04 |
| v1 | Geometry + human trajectory + object trajectory | **52.36%** | 52.06% | **+0.30** | 53.36% | 53.08% | +0.28 |
| v5 | Geometry + unordered object summary | 51.67% | 52.06% | -0.39 | 53.22% | 53.08% | +0.14 |
| v4 | v1 + human torso orientation | 51.85% | 52.06% | -0.21 | **53.60%** | 53.08% | **+0.52** |

## 4. Random-Start Results

| Version | 1-move Policy | Random | Diff | 2-move Policy | Random | Diff |
|---|---:|---:|---:|---:|---:|---:|
| geometry_only | 51.42% | 51.88% | -0.45 | 52.97% | 53.09% | -0.13 |
| v6 / person_only | 52.05% | 51.88% | +0.17 | **53.48%** | 53.09% | **+0.39** |
| object_only | 52.31% | 51.88% | **+0.44** | 53.14% | 53.09% | +0.05 |
| v1 | **52.34%** | 51.88% | **+0.46** | 53.43% | 53.09% | +0.34 |
| v5 | 51.60% | 51.88% | -0.28 | 53.32% | 53.09% | +0.23 |
| v4 | 51.91% | 51.88% | +0.03 | **53.63%** | 53.09% | **+0.53** |

The `±` for Policy is not included in the summary tables; the raw JSON records the standard deviation across the 3 training checkpoints and across the 30 evaluation seeds.

## 5. Conclusions

### 5.1 Geometry alone is not sufficient

Geometry-only is below Random under both fixed-start and random-start protocols. Candidate geometry is a necessary input, but by itself it cannot judge which candidate view has higher recognition value.

### 5.2 One move benefits mainly from the object trajectory

Object-only already exceeds Random; v1's improvement is slightly higher than object-only. Thus for one move, object presence, position, confidence, and position relative to the human are the main effective information. The unordered summary in v5 actually degrades, indicating that the class/slot structure of the 50 slots in the current data still carries useful information.

### 5.3 Two moves benefit mainly from the human trajectory

Person-only (v6) is stronger than object-only for two moves, and stably exceeds Random under both fixed-start and random-start protocols. The temporal human center/velocity helps plan the second transition by continuing from the already observed trajectory.

### 5.4 A single branch of v1 cannot simply be removed permanently

v1 is best for one move; for two moves v6 is slightly higher than v1, and v4 (with human orientation added) is best. Therefore the reasonable pruning is not one-size-fits-all:

- **One move only**: prefer keeping geometry + raw object trajectory; if the highest average accuracy is the goal, keep full v1;
- **Two moves**: prefer keeping geometry + human trajectory, or use v4 with human orientation added; the extra gain from the object branch is small;
- **Single general-purpose model**: keep v1 for now, to avoid losing one-move performance over a ~0.05–0.14 percentage-point fluctuation for two moves.

In other words, this ablation supports the complementarity of the human trajectory and object trajectory branches, but does not support removing either branch across all move counts.

## 6. Result Files

- New training code: `rl/multistep_angle_object_trajectory_ablation.py`
- Fixed-start summary: `work_dir/trajectory_ablation_old_split/fixed_30seeds/`
- Random-start summary: `work_dir/trajectory_ablation_old_split/random_start_30seeds/`
- New object-only training: `work_dir/trajectory_ablation_old_split/object_only/`
- New geometry-only training: `work_dir/trajectory_ablation_old_split/geometry_only/`
