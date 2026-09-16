# Two-Stage Experiment Plan: Dual-Robot Geometric Coordination + Synthetic-Occlusion Reinforcement Learning

## Overall Sequence

Execute strictly in the order below. Do not proceed in parallel, and do not automatically start large-scale RL for Stage 2 before Stage 1 results are complete:

```text
Experiment 1
Dual-robot / dual-view geometric coordination
Preferred Orthogonal (non-learning)
        ↓
Complete equal-budget benchmark + bootstrap
        ↓
Experiment 2
2D View-Consistent Synthetic Occlusion
        ↓
Run active-view diagnostic first
        ↓
Only after the diagnostic succeeds
        ↓
Train RL
```

The two experiments answer different questions:

- **Experiment 1**: Under the original ETRI with almost no occlusion, is a simple class-agnostic geometrically complementary dual view already effective enough?
- **Experiment 2**: If view-consistent occlusion is artificially introduced so that "moving" genuinely changes observability, can RL finally learn a meaningful occlusion-avoidance / occlusion-bypass viewpoint strategy?

---

# Experiment 1 — Dual-Robot Preferred Orthogonal Cooperative Perception

## 1. Experiment Objectives

Use ETRI synchronized multi-view data to simulate two robots offline:

```text
Robot A:
Determine the observation position first, as the anchor robot

Robot B:
Based on the relative viewing direction between Robot A and the human
Use the Preferred Orthogonal geometric rule
Select the most complementary second observation position

Robot A + Robot B:
Simultaneously observe the same temporal window
Run HAR simultaneously
Finally perform dual-view fusion
```

The active placement algorithm in this experiment **does not use RL and does not train a viewpoint policy**.

The active viewpoint selection part is:

```text
class-agnostic
human-centric
geometry-based
non-learning viewpoint coordination
```

The VPOCLIP recognizer remains learning-based.

---

## 2. Key Scientific Questions

Two distinct gains must be separated:

### Extra-sensor gain

```text
Dual-Robot Random Pair
-
Single-Robot Fixed
```

This gain mainly comes from "having one more camera/robot".

### Coordination gain

```text
Preferred Orthogonal Pair
-
Dual-Robot Random Pair
```

This is the contribution of the geometric coordination algorithm itself.

The final report must not only emphasize:

```text
Ours - Single Fixed
```

The following must also be reported:

```text
Ours - Random Pair
Ours - Fixed Pair
```

---

## 3. Data and Protocol

Keep the existing protocol:

```text
Seen train
Seen validation / subject-held-out validation
Strict true unseen test
```

Current true unseen:

```text
[0, 2, 26, 34, 50]
```

Important:

```text
It is forbidden to tune on true unseen:
- camera angle
- orthogonal target angle
- fusion weight
- threshold
- motion-cost weight
- model checkpoint
```

True unseen is used only for the final locked evaluation.

---

## 4. Meaning of the ETRI Dual-Robot Offline Simulation

ETRI raw data is recorded by fixed synchronized multi-camera capture, so do not claim in the paper/report that real dual-robot motion experiments have already been performed.

The current stage should accurately be called:

```text
dual-robot sensing proxy
or
dual-agent synchronized multi-view simulation using ETRI cameras
```

Each ETRI camera view represents a discrete observation position reachable by a robot.

If real robots become available later, perform physical deployment validation.

---

## 5. Camera Relative Angle Configuration

If ETRI has no camera calibration, do not fabricate intrinsics/extrinsics.

Create a configuration file:

```text
camera_view_angles.yaml
```

For example:

```yaml
view0: 0
view1: 90
view2: 180
view3: 270
```

But the specific mapping here must be determined based on:

```text
ETRI camera ordering
Existing metadata
Manual quick inspection of a few synchronized samples
```

Determine it from these sources.

If only the discrete surround order is known, the following may also be used:

```text
0 / 90 / 180 / 270
```

as pseudo-angles.

In the report, explicitly call it:

```text
relative camera-view angle
```

Do not call it a calibrated world pose.

---

## 6. Robot A — Anchor Placement

For fair comparison with previous results, the main experiment uses:

```text
Robot A = fixed View0
```

Additionally, run robustness tests:

```text
Robot A = random valid anchor
```

But the main table should prioritize fixed View0.

---

## 7. Robot B — Preferred Orthogonal

Let the direction of Robot A relative to the human be:

\[
\theta_A
\]

Candidate Robot B positions:

\[
\theta_j
\]

Compute the circular angular separation:

\[
\Delta\theta_j
=
\min(
|\theta_j-\theta_A|,
360-|\theta_j-\theta_A|
)
\]

The Preferred Orthogonal selection:

\[
j^*
=
\arg\min_j
|\Delta\theta_j-\theta^*|
\]

Default:

\[
\theta^*=90^\circ
\]

If multiple candidates tie:

1. Prefer the one with lower movement cost;
2. If no movement cost is available, use a fixed deterministic tie-break;
3. It is not allowed to inspect HAR GT or future candidate logits for tie-breaking.

---

## 8. Optional Geometric Variants

Run only a small-scale ablation; do not over-search:

```text
Target angle:
60°
90°
120°
```

Parameters may only be selected on validation.

The main method should prefer to keep:

```text
90° Preferred Orthogonal
```

to keep the method simple and interpretable.

---

## 9. Dual-View HAR Fusion

To isolate the contribution of "placement" alone, do not over-complicate the first version of fusion.

Main baseline:

\[
z_{fused}
=
\frac{z_A+z_B}{2}
\]

where:

```text
z_A = Robot A logits
z_B = Robot B logits
```

Optionally test:

```text
max confidence fusion
weighted logits fusion
```

But fusion parameters must not be tuned on true unseen.

For now, do not introduce a new large cross-view network; otherwise it is impossible to tell whether the gains come from placement or fusion.

---

## 10. Baselines That Experiment 1 Must Compare

Must include at least:

### 1. Single-Robot Fixed

```text
Robot A at View0
1 view
```

### 2. Single-Robot Random-2

Simulate a single robot:

```text
View0
+
A random second view
```

Used to align with previous Random-2 results.

### 3. Dual-Robot Fixed Pair

Fix a camera pair, for example:

```text
View0 + predefined View1
```

Used to measure the gain of a "fixed dual-robot" setup by itself.

### 4. Dual-Robot Random Pair

```text
Robot A = View0
Robot B = random legal candidate
```

This is the most important equal-budget baseline.

### 5. Dual-Robot Maximum Separation

Select the view with the largest angular difference from Robot A.

Used to distinguish:

```text
90° complementary
vs
simply farther is better
```

### 6. Dual-Robot Preferred Orthogonal

The proposed method of this experiment.

### 7. All Valid Views

Fuse all views.

### 8. Oracle-2

Use GT to offline-select the optimal second view.

Can only serve as an upper bound.

---

## 11. Experiment 1 Main Results Table

Codex must finally generate:

| Method | # Views | Seen Top-1 | True Unseen Top-1 | Δ vs Single Fixed | Δ vs Random Pair | 95% CI vs Random Pair |
|---|---:|---:|---:|---:|---:|---:|
| Single Fixed | 1 |  |  |  |  |  |
| Single Random-2 | 2 |  |  |  |  |  |
| Dual Fixed Pair | 2 |  |  |  |  |  |
| Dual Random Pair | 2 |  |  |  |  |  |
| Max Separation Pair | 2 |  |  |  |  |  |
| Preferred Orthogonal Pair | 2 |  |  |  |  |  |
| All Views | 4 |  |  |  |  |  |
| Oracle-2 | 2 |  |  |  |  |  |

---

## 12. Experiment 1 Statistical Tests

For:

```text
Preferred Orthogonal
vs
Dual Random Pair
```

A paired bootstrap is required:

```text
>= 5000 bootstrap samples
```

Report:

```text
accuracy difference
95% CI
```

Also run:

```text
Preferred Orthogonal
vs
Dual Fixed Pair
```

---

## 13. Per-Class Analysis

Report per class on true unseen `[0,2,26,34,50]`:

```text
Single Fixed
Random Pair
Preferred Orthogonal
Oracle-2
```

This prevents the overall improvement from being driven entirely by a single action class.

---

## 14. Experiment 1 Go / No-Go

### Positive result

If the following appears:

```text
Preferred Orthogonal > Random Pair
```

And:

```text
Most splits / classes are positive
The CI shifts clearly in the positive direction
```

then geometry-based dual-robot coordination is worth continuing as the main method.

### Weak but useful result

If:

```text
Preferred Orthogonal is only 0.5~1 pp above Random Pair
```

but:

```text
stable
and lower movement / placement cost
```

keep it anyway, and shift the paper's focus to:

```text
accuracy-cost tradeoff
```

### Negative result

If:

```text
Preferred Orthogonal ~= Random Pair
```

and there is no stable advantage on any split, do not package it as a "learning success".

Proceed to Experiment 2 to verify whether the original ETRI lacks active-perception pressure.

---

# Experiment 2 — View-Consistent Synthetic Occlusion + RL

## 15. Experiment Objectives

Verify the following hypothesis:

> The original ETRI environment has little occlusion (to clearly observe elderly behavior), so the action-value differences between viewpoints are insufficient, causing previous RL / NBV / VoI methods to struggle to learn effective movement policies.

This experiment artificially generates:

```text
view-dependent
multi-view consistent
human-centric synthetic occlusion
```

such that:

```text
Some views are occluded
Lateral movement reduces the occlusion
Movement in the opposite direction increases the occlusion
```

Then re-examine:

```text
Oracle gap
recoverable ratio
visibility-to-HAR relationship
RL performance
```

---

# 16. Do Not Perform 3D Calibration Directly in This Experiment

If the current ETRI data lacks camera intrinsics/extrinsics:

```text
Do not attempt to fabricate calibration
Do not start with 3D reconstruction
Do not start with NeRF / Gaussian Splatting
```

The first version uses:

```text
2D top-down pseudo-world
+
relative camera angles
+
virtual occluder
```

---

# 17. Top-Down Pseudo-World

Define the human center:

\[
H=(0,0)
\]

Four cameras:

\[
C_i=
R[
\cos\theta_i,
\sin\theta_i
]
\]

where:

```text
theta_i
```

comes from the `camera_view_angles.yaml` of Experiment 1.

Real metric positions are not needed.

For example:

```text
Human = origin
Camera radius R = 1.0
```

---

# 18. Virtual Occluder

Sample only once per synchronized multi-view clip:

```text
occluder_angle
occluder_distance
occluder_width
severity
target_region
```

Then keep it fixed for the entire clip.

Example:

```text
occluder_angle = -30°
occluder_distance = 0.4
severity = medium
target_region = upper_body
```

All views of the same sample must share the same virtual occluder.

Independent random masks per view are forbidden.

---

# 19. Computing View-Dependent Occlusion

The first version may use line-of-sight distance.

For the line-of-sight segment from camera \(C_i\) to the human \(H\):

\[
L_i=C_i\rightarrow H
\]

Compute the shortest distance from the virtual occluder \(O\) to this segment:

\[
d_i = distance(O,L_i)
\]

Define the occlusion severity:

\[
s_i
=
s_{max}
\exp
\left(
-\frac{d_i^2}{2\sigma_d^2}
\right)
\]

Also check whether the occluder lies between:

```text
camera -> human
```

the camera and the human.

If it is not in front of the line of sight:

```text
s_i = 0
```

---

## 20. Occlusion Direction

Use signed lateral distance / 2D cross product to determine:

```text
the occluder is on the left of the human in the current camera view
or
on the right
```

Then the mask enters the human bbox from the corresponding side.

Target effect:

```text
Front view:
moderate/heavy occlusion

Left view:
more severe

Right view:
mild or completely gone
```

This must arise from the geometry of the same virtual occluder, not from hard-coded per-camera masks.

---

# 21. Human-Centric 2D Mask

Prefer using existing:

```text
person bbox
or
2D skeleton
```

to obtain the per-frame human bbox:

```text
x1(t), y1(t), x2(t), y2(t)
```

If no person bbox is available:

```text
Generate the bbox
from the min/max of valid 2D keypoints
then expand it by a 5~10% margin
```

mask width:

```text
mask_width
=
severity * human_bbox_width
```

mask height in the first version:

```text
0.6 ~ 0.8 * human_bbox_height
```

Prefer occluding:

```text
upper body
hands
face / torso interaction region
```

rather than random background.

---

# 22. Temporal Consistency

For the same clip:

```text
occluder world parameters fixed
```

Per frame, update only the mask position on the image plane based on the person bbox.

To avoid bbox jitter:

```text
Apply EMA smoothing to the bbox center / size
```

Randomly changing the mask per frame is forbidden.

---

# 23. Mask Appearance

By stage:

### Debug

```text
gray rectangle
```

### Main synthetic benchmark

Recommended:

```text
blurred / textured rectangle
```

Avoid pure black blocks that create an overly strong synthetic artifact.

You may crop texture from non-human background and fill the mask with it.

Do not do complex object cutouts in the first round.

---

# 24. Do Not Re-encode All MP4s

Prefer:

```text
on-the-fly masking
```

In the Dataset / dataloader:

```python
frames = load_clip(...)
frames = apply_view_consistent_occlusion(...)
model_input = preprocess(frames)
```

Do not save large numbers of new videos by default.

You may cache only:

```text
sample_id
occluder_angle
occluder_distance
severity
target_region
random_seed
```

Ensure all methods use exactly the same synthetic episodes.

---

# 25. Extremely Important: Avoid "Unoccluded Auxiliary-Modality Leakage"

Check in the current VPOCLIP pipeline:

```text
RGB
Skeleton
Object
```

whether they come from the original unoccluded cache.

If RGB is masked but skeleton / object still come from the original clear video, information leakage occurs:

```text
Visually occluded
but the model still sees the full human through unoccluded skeleton/object
```

This is not allowed.

Codex must first inspect the current data flow, then choose one of the following consistent implementations:

### Preferred

Apply the mask before all feature extraction:

```text
masked frame
-> pose/object extraction
-> VPOCLIP
```

Regenerate the occluded inputs.

### If pose/object reconstruction is too costly

Based on the mask region:

```text
keypoints falling inside the mask -> missing / zero / invalid
objects largely covered by the mask -> drop / invalid
```

The complete unoccluded pose/object must not be used.

### If consistent corruption is not feasible

First run only an explicitly labeled:

```text
RGB-only synthetic occlusion diagnostic
```

Do not mislabel its results as a full multimodal VPOCLIP result.

---

# 26. Occlusion Severity Levels

Use exactly four levels:

```text
None
Mild
Medium
Heavy
```

Suggested initial ranges:

```text
Mild:
15~30% target-region occlusion

Medium:
30~50%

Heavy:
50~70%
```

Specific parameters must be confirmed only on seen validation.

---

# 27. Diagnostic Must Run Before RL

Before any PPO/RL training, for each of:

```text
None
Mild
Medium
Heavy
```

compute:

```text
View0
Random-2
Preferred Orthogonal-2
Oracle-2
Recoverable ratio
```

Main table:

| Occlusion | View0 | Random-2 | Orthogonal-2 | Oracle-2 | Oracle-Random Gap | Recoverable Ratio |
|---|---:|---:|---:|---:|---:|---:|
| None | | | | | | |
| Mild | | | | | | |
| Medium | | | | | | |
| Heavy | | | | | | |

---

# 28. Active-View Signal Diagnostic

Because the synthetic generator knows the actual occlusion level of each candidate, define:

```text
least_occluded_candidate
```

Then compute:

```text
P(least_occluded_candidate == HAR_best_candidate)
```

where:

```text
HAR_best_candidate
```

is used only for the offline diagnostic, determined by GT utility.

With 3 candidates, random action agreement is about:

```text
33.3%
```

After synthetic occlusion, we hope that:

```text
least-occluded vs HAR-best agreement
is clearly > random
```

For example, reaching:

```text
50%+
```

More ideally:

```text
60~80%
```

---

# 29. Experiment 2 RL Go / No-Go Gate

Start RL only if most of the following conditions are met:

```text
1. The Oracle-Random gap clearly widens compared to None
2. The recoverable ratio clearly increases
3. The agreement between the least-occluded candidate and the HAR-best candidate is clearly higher than random
4. The Orthogonal / low-occlusion view shows a stable advantage under synthetic occlusion
```

If none of these appear:

```text
Do not train PPO
```

This means the synthetic mask did not create reasonable active-view pressure; fix the occlusion generator first.

---

# 30. RL Problem Formulation

If the diagnostic passes, construct a truly sequential RL problem.

### Episode

```text
start at one view
observe masked clip
choose:
    STOP
    move to another valid view
observe new masked synchronized view
optionally move again
STOP
```

At most:

```text
2 moves
```

Do not use an infinite horizon in the first version.

---

# 31. RL State

Only information already observed currently or historically is allowed:

```text
current VPOCLIP feature / logits
current entropy
current top1-top2 margin

current observed occlusion ratio
current observed mask side
current human bbox / visible keypoints

current camera relative angle
visited-view mask
history fused logits
number of views used
movement cost
```

Note:

```text
current observed occlusion ratio / side
```

can be computed from the current synthetic mask or the current image-derived mask, so it is causally observable information.

---

# 32. Inputs Strictly Forbidden for RL

Forbidden:

```text
GT action label
future candidate RGB
future candidate logits
future candidate VPOCLIP feature

future candidate occlusion severity
future candidate visibility
latent occluder_angle
latent occluder_position

HAR Oracle action
```

Special attention:

**The hidden occluder_angle of the synthetic generator must not be given directly to the policy.**

Otherwise the policy is reading simulator ground truth, not performing true active vision.

---

# 33. RL Action

With four-view data:

```text
STOP
move_to_view0
move_to_view1
move_to_view2
move_to_view3
```

Already-visited views:

```text
mask as illegal
```

The current view:

```text
Cannot be re-selected in place
```

---

# 34. RL Reward

Recommended for the first version:

\[
r_t
=
\alpha\Delta HAR_t
+
\beta\Delta Obs_t
-
\lambda C_{move}
-
\mu C_{view}
\]

where:

### HAR evidence gain

When training on seen classes:

\[
\Delta HAR_t
=
Margin_{t+1}^{GT}
-
Margin_t^{GT}
\]

Used only for reward, not as state.

### Observable quality gain

May use:

\[
\Delta Obs
=
Occ_t-Occ_{t+1}
\]

i.e., occlusion reduction.

But do not let the visibility reward completely dominate.

Suggested starting values:

```text
alpha = 1.0
beta = 0.2
lambda = 0.05
mu = 0.02
```

Start from this magnitude, then adjust within a limited range on validation.

Do not tune the reward on true unseen.

---

# 35. Terminal Reward

After STOP:

```text
correct final HAR:
+ positive reward

incorrect:
0 or small negative
```

Keep the first version simple.

---

# 36. RL Algorithm

If the existing project already has PPO infrastructure:

```text
Prefer to continue with PPO
```

But start with a small network:

```text
MLP 64-64
```

Then compare at most:

```text
128-128
```

Do not jump directly to a large Transformer.

The goal of the current experiment is to verify:

```text
occlusion makes active movement learnable
```

not to prove that a larger network is stronger.

---

# 37. RL Training Data

Train only on seen classes.

Randomly sample the synthetic occluder per epoch / episode:

```text
direction
distance
severity
target region
```

This forms domain randomization.

Mix during training:

```text
None
Mild
Medium
Heavy
```

For example:

```text
10% None
30% Mild
35% Medium
25% Heavy
```

The specific proportions may only be tuned on validation.

---

# 38. Prevent RL from Learning a Camera-ID Shortcut

Must:

```text
random start view
random occluder direction
random severity
random subject/sample order
```

Do not always have:

```text
View0 occluded
View2 clearest
```

Otherwise the policy will only memorize:

```text
View0 -> View2
```

instead of learning to "avoid occlusion".

---

# 39. RL Baselines

Under synthetic occlusion, compare at least:

```text
Fixed View
Random Move
Preferred Orthogonal
Greedy Current-Visibility heuristic
PPO / RL
Oracle
```

### Greedy Current-Visibility heuristic

May only use the current observation:

```text
mask side
camera geometry
```

Use a simple rule to choose a lateral move.

This is an important baseline for judging whether RL truly outperforms hand-designed avoidance.

---

# 40. RL Main Results Table

| Occlusion | Fixed | Random | Orthogonal | Visibility Heuristic | RL | Oracle |
|---|---:|---:|---:|---:|---:|---:|
| None | | | | | | |
| Mild | | | | | | |
| Medium | | | | | | |
| Heavy | | | | | | |

Also report:

```text
average views
average moves
average occlusion after move
recoverable success rate
```

---

# 41. True Unseen Testing

Finally, on locked true unseen:

```text
[0,2,26,34,50]
```

test synthetic-occlusion transfer.

Important:

```text
Same samples
Same occluder seed
Same severity
```

All methods must undergo paired evaluation.

---

# 42. RL Success Criteria

The ideal result is not judged only on:

```text
RL > Fixed
```

but on:

```text
RL > Random
RL > Preferred Orthogonal
RL > simple visibility heuristic
```

Especially under:

```text
Medium
Heavy
```

occlusion.

If:

```text
RL is only higher than Fixed
but ~= Random / heuristic
```

then one cannot claim that RL learned an effective active-viewpoint policy.

---

# 43. Final Joint Table for Both Experiments

Finally, generate a summary table:

| Setting | Method | True Unseen Top-1 | Δ vs Equal-Budget Random | Avg Views | Notes |
|---|---|---:|---:|---:|---|
| Original ETRI | Single Fixed | | | 1 | |
| Original ETRI | Dual Random Pair | | | 2 | |
| Original ETRI | Dual Preferred Orthogonal | | | 2 | |
| Synthetic Mild | Random | | | | |
| Synthetic Mild | RL | | | | |
| Synthetic Medium | Random | | | | |
| Synthetic Medium | RL | | | | |
| Synthetic Heavy | Random | | | | |
| Synthetic Heavy | RL | | | | |

---

# 44. Scientific Questions Codex Must Finally Answer

Experiment 1:

```text
1. How much does dual-robot synchronized dual-view improve over a single robot?
2. How much of that gain comes from "one more sensor"?
3. How much does Preferred Orthogonal actually improve over the equal-budget Random Pair?
4. Is this improvement stable across unseen classes?
```

Experiment 2:

```text
1. Does view-consistent synthetic occlusion widen the Oracle-Random gap?
2. Does the recoverable ratio increase with occlusion?
3. Are less-occluded candidates more likely to be the HAR-best candidate?
4. Does RL learn to move in directions that reduce occlusion?
5. Does RL outperform Random / Orthogonal / heuristic on true unseen actions?
6. Can the failure of RL in the original ETRI be partly explained by insufficient active-perception pressure?
```

---

# 45. Automatic Execution Order

Codex executes strictly:

```text
PHASE 1
Check the data view mapping
Build the relative camera angle config

PHASE 2
Run Experiment 1
Do not train a viewpoint model

PHASE 3
Output Experiment 1 benchmark + bootstrap

PHASE 4
Implement the 2D top-down virtual occluder
First visualize several synchronized samples
Confirm:
occlusion changes across front / left / right follow the geometry

PHASE 5
Check multimodal leakage
Ensure masked RGB is not paired with completely unoccluded pose/object

PHASE 6
Run only the synthetic occlusion diagnostic
Do not train RL

PHASE 7
Check the RL Go / No-Go Gate

PHASE 8
Train PPO only if the gate passes

PHASE 9
Output the synthetic RL benchmark

PHASE 10
Generate the joint summary table and logs
```

---

# 46. Suggested Output Directories

Create under the repository root:

```text
work_dir/
  exp01_dual_robot_orthogonal/
    config/
    results/
    figures/
    logs/

  exp02_synthetic_occlusion_rl/
    config/
    masks_debug/
    diagnostics/
    checkpoints/
    results/
    figures/
    logs/
```

Save at least the following at the end:

```text
summary.json
per_sample_predictions.csv
per_class_results.csv
bootstrap_results.json
config.yaml
run.log
```

Experiment 2 additionally saves:

```text
synthetic_occlusion_config.json
occlusion_diagnostic.csv
rl_training_curve.csv
```

---

# 47. Prohibited Actions

Codex must not:

```text
Tune parameters on true unseen
Feed Oracle information into the policy
Feed future candidate information into the policy
Use independent random masks per view
Use the synthetic occluder's latent world position directly as policy state
Automatically train RL after the gate fails
Change the VPOCLIP checkpoint without authorization to improve the numbers
Describe ETRI fixed-camera results as real dual-robot experiments
```

---

# 48. Final Decision Logic

```text
Experiment 1 succeeds:
    Keep Dual-Robot Preferred Orthogonal
    as the simple, robust main method for the original occlusion-free environment

Experiment 1 is weak:
    Do not reject it immediately
    Proceed to Experiment 2

Experiment 2 diagnostic succeeds + RL succeeds:
    This shows RL needs sufficiently strong active-perception pressure
    A complete story of "occlusion-free geometric coordination + learned active viewpoint under occlusion" can be formed

Experiment 2 diagnostic succeeds + RL fails:
    Occlusion does create an active-view opportunity
    But the current RL/state is still insufficient
    Compare geometry / heuristic methods; do not force a claim of RL success

Experiment 2 diagnostic fails:
    The synthetic occlusion design did not create a reasonable active-view problem
    Stop RL and fix the occlusion simulator first
```

---

# 49. Final Research Narrative

If both experiments yield reasonable results, the following story can be formed:

> In the original ETRI setting, where occlusion is limited, a simple class-agnostic complementary dual-view formation is competitive and robust for unseen-action recognition. When viewpoint-dependent occlusion is introduced, active movement becomes more consequential; under this setting, we test whether an RL policy can learn to move toward views that recover task-relevant evidence.

Note:

- Original ETRI results demonstrate the value of simple geometric coordination;
- Synthetic occlusion is used to study whether active viewpoint learning is necessary when real occlusion pressure exists;
- Do not treat synthetic occlusion as equivalent to real scenes;
- For a future robotics venue submission, it is best to add a small number of real-robot + real-occluder experiments later.
