# Improved Plan: Environment-Conditioned Active View Policy with View-Consistent Synthetic Occlusion

Suggested version: `multistep_environment_occlusion_policy_v2`

## 0. Goals for This Round

Do not keep defining the problem this round as:

```text
"What action class is this -> which viewpoint should we move to?"
```

The new core problem is:

```text
"Where is the human now, how are they moving, where are the occluders/objects,
and where is the current camera
-> which direction of movement yields a more complete, more complementary observation?"
```

The policy therefore learns:

\[
\boxed{
\text{human-environment spatial state}
\rightarrow
\text{view acquisition strategy}
}
\]

rather than:

\[
\boxed{
\text{action semantics}
\rightarrow
\text{view preference}
}
\]

The final goal remains unseen/open-vocabulary HAR, but the viewpoint policy must be as **class-agnostic** as possible.

---

# 1. Why Synthetic Occlusion Is Worth Doing Now

The full ETRI dataset was recorded specifically to clearly capture elderly behavior, so severe occlusion is rare in the real videos.

Previous experiments repeatedly showed:

```text
small gap between Random and the learned policy
a large fraction of samples are view-insensitive
Oracle is clearly above Random, but the best viewpoint is hard to predict from the current observation
```

A reasonable hypothesis is:

\[
\boxed{
\text{the original ETRI lacks sufficiently strong active-perception pressure}
}
\]

In other words, in a large fraction of episodes:

```text
switch to the left
switch to the right
switch to the back
```

all reveal roughly the same information.

Thus:

\[
Q(s,a_1)\approx Q(s,a_2)\approx Q(s,a_3)
\]

and the policy receives little discriminative supervision.

This round uses **view-consistent synthetic occlusion** to artificially create:

```text
occlusion at the current viewpoint
reduced occlusion from a side viewpoint
hand/object/body evidence recovered from another direction
```

and thereby tests:

> If the environment genuinely makes observability depend on the direction of movement, can the current human/object/geometry policy learn a transferable spatial strategy?

---

# 2. Most Important Principles for This Round

## 2.1 Do Not Use Plain Random Erasing

Forbidden: only doing

```text
random black square
random cutout
independently random mask per view
```

Because this augmentation does not establish:

\[
\text{camera motion}
\rightarrow
\text{occlusion change}
\]

as a causal relationship.

If the View0 mask is fully independent of View1/View2/View3, the policy cannot infer which direction to move from the current state.

---

## 2.2 View-Consistent Occlusion Is Required

Sample only one virtual occluder per episode:

```text
occluder bearing
occluder width
occluder height
occluder strength
target vertical region
render style
```

Then generate 2D occlusion of varying degree according to each camera's true `relative_angle` with respect to the human.

Core idea:

\[
\boxed{
\text{the same virtual occluder produces different occlusion under different cameras}
}
\]

---

# 3. Keep the Core Structure of the Current Trajectory-Conditioned Policy

Already present in v1:

```text
person trajectory
object trajectory
sample-level camera relative angle
candidate-current delta angle
history of observed views
action mask
```

Do not overturn these designs this round.

Keep enforcing:

```text
VPOCLIP frozen
policy does not take action class ID as input
policy does not take the GT label as input
policy does not take future candidate RGB as input
policy does not take future candidate logits as input
policy does not take future candidate object/pose as input
```

v2 mainly adds:

```text
explicit observed occlusion/environment state
+
synthetic view-consistent occlusion benchmark
+
occlusion-aware training utility
```

---

# 4. Phase 0: Build the Synthetic Occlusion Benchmark First; Do Not Train the Policy

This is the most important step.

First answer:

> Does synthetic occlusion genuinely increase the value of viewpoint selection?

If not, do not train any new network.

---

# 5. Use the Existing Sample-Level Camera Bearings

Each sample already has:

```text
view_geometry[..., 0:2]
=
[sin(relative_angle), cos(relative_angle)]
```

Do not use fixed camera slot semantics.

For each recording, first recover:

\[
\theta_v = \operatorname{atan2}(\sin\theta_v,\cos\theta_v)
\]

to obtain the human-relative bearing of each valid view.

Define the virtual occluder also as a human-relative bearing:

\[
\theta_{occ}
\]

Sampled once per episode.

---

# 6. First Version: 2D Pseudo-Geometric Occlusion

Full 3D reconstruction is not required at this stage.

Use:

```text
human-relative camera angle
person bbox / torso center
pose trajectory
object trajectory
```

to build an approximately geometry-consistent occlusion model.

---

# 7. Occlusion Severity

For camera \(v\), define:

\[
\Delta\theta_v
=
wrap(\theta_v-\theta_{occ})
\]

Occlusion strength:

\[
r_v
=
r_{max}
\exp
\left(
-\frac{\Delta\theta_v^2}{2\sigma_{occ}^2}
\right)
\]

Interpretation:

```text
camera bearing close to the occluder bearing
-> severe occlusion

camera turned to the side
-> moderate occlusion

camera bearing far from the occluder bearing
-> weak/no occlusion
```

Suggested for the first version:

```text
sigma_occ = 35° ~ 55°
```

Do not fix a single value directly; select it on pseudo-unseen validation.

---

# 8. Occlusion Level

Build at least a three-level benchmark:

### Mild

```text
r_max ≈ 0.25 ~ 0.35
```

### Medium

```text
r_max ≈ 0.45 ~ 0.55
```

### Heavy

```text
r_max ≈ 0.65 ~ 0.75
```

Do not interpret these ratios simply as a fraction of the whole image area.

Define them preferably as:

```text
the occluded fraction of the person / interaction region
```

---

# 9. Masks Should Not Mainly Cover the Background

The target region must be related to human observation.

Implement at least 3 types of occlusion target.

## A. Torso-centered

Based on COCO17:

```text
shoulders: 5, 6
hips:      11, 12
```

construct a torso bbox.

Suitable for simulating:

```text
cabinet
table edge
door frame
furniture
```

Occludes the middle of the body.

---

## B. Upper-body / hand region

Use:

```text
shoulder
elbow
wrist
face/head proxy
```

Prioritize occluding:

```text
hand
face
upper torso
```

because many HAR actions depend on these regions.

---

## C. Hand-object interaction region

Use the current object trajectory:

```text
[presence, x, y, confidence]
```

For each frame, find the object that is:

```text
closest to the left/right wrist
with sufficiently high confidence
```

Construct a joint region of:

```text
hand + nearest object
```

Note:

```text
the GT action class must not be used to decide where to mask
```

---

# 10. The Mask's Horizontal Offset Must Also Vary with Viewpoint

Do not only vary the mask area.

Define the mask horizontal offset relative to the person center:

\[
x_{offset,v}
=
k_x \sin(\Delta\theta_v)
\]

Thus:

```text
facing the occluder:
mask stays near the person center

camera moves to one side:
the mask gradually shifts out of the key human region in the image
```

mask width:

\[
w_v
=
w_{base}\cdot r_v
\]

mask height:

\[
h_v
=
h_{base}\cdot
(0.6+0.4r_v)
\]

The first version does not require strict physical projection, but it must ensure:

> mask variation is determined by a single occluder bearing plus camera bearing, not independently random per view.

---

# 11. Temporal Consistency

For the 13 frames of the same view:

```text
the mask must not be re-randomized per frame
```

The occluder parameters are fixed for the whole episode:

```text
theta_occ
width_base
height_base
render_style
```

As the person moves across the 13 frames, the mask may be updated smoothly based on the torso / interaction region.

Requirements:

```text
continuous mask center trajectory
continuous mask size trajectory
no frame-to-frame flickering
```

Add EMA:

\[
M_t =
\beta M_{t-1}
+
(1-\beta)\hat M_t
\]

First version:

```text
beta ≈ 0.7
```

---

# 12. Render Style

Do not use plain black rectangles only.

Otherwise VPOCLIP / the policy may learn synthetic artifacts.

Randomize at least:

```text
neutral gray
local blur
background-color fill
texture patch
```

Randomize the renderer during training.

At test time, evaluate at least:

```text
seen renderer
held-out renderer
```

Check whether the policy merely memorizes the mask appearance.

---

# 13. Do Not Destroy All Views

After generating an occlusion episode, verify:

```text
at least one valid view still has fairly clear key human regions
```

Otherwise the episode is:

```text
unrecoverable by construction
```

and provides no meaningful target for active-view learning.

Suggested constraint:

\[
\min_v r_v < r_{clear}
\]

For example:

```text
at least one view has task-relevant occlusion < 20%
```

---

# 14. Phase 0 Must Re-run the VPOCLIP Cache

The mask must actually be applied to the input video and re-encoded by the frozen VPOCLIP.

Do not:

```text
only modify the policy feature
while HAR logits still come from un-occluded videos
```

Otherwise reward/utility is inconsistent with the synthetic environment.

For every:

```text
sample
view
occlusion seed
```

regenerate:

```text
masked video / frames
VPOCLIP logits
pose if possible
object tracks if possible
```

---

# 15. How to Handle Pose / Object

## Preferred

If compute allows:

```text
re-run the pose / object detector on masked frames
```

The observed state then naturally reflects:

```text
keypoint dropout caused by occlusion
reduced object confidence
degraded trajectory
```

This is the most realistic version.

---

## Minimum viable version

If re-running detectors is too expensive:

keep using the original pose/object trajectory, but additionally include:

```text
synthetic mask descriptors
```

explicitly in the policy state.

State explicitly in the report:

> pose/object trajectories are privileged clean sensor tracks in the first synthetic benchmark.

Then build the masked-detector version later.

---

# 16. New Occlusion / Environment State

For each observed view and each time step, store:

```text
mask_area_ratio
mask_person_overlap
mask_torso_overlap
mask_upper_body_overlap
mask_left_hand_overlap
mask_right_hand_overlap
mask_interaction_overlap

mask_center_x_relative_to_person
mask_center_y_relative_to_person
mask_width_relative_to_person
mask_height_relative_to_person
```

Suggested shape:

```text
occlusion_trajectory: [13, 10~12]
```

---

# 17. Future Candidate Occlusion Must Not Be Input

This is a crucial causal constraint.

The policy can only see in the current state:

```text
mask / occlusion descriptors of already visited views
```

It must not directly receive:

```text
true mask_area_ratio of candidate views
true visibility of candidate views
future occlusion of candidate views
```

Otherwise it amounts to giving the policy the answer.

---

# 18. Optional Map-Aware Variant

If you later want to simulate a robot that already has an environment map, build a separate second version.

The map-aware state may add:

```text
estimated obstacle bearing
estimated obstacle distance
obstacle width
```

But report separately:

```text
No-map policy
Map-aware policy
```

Prioritize No-map for the main results to avoid privileged-information concerns.

---

# 19. v2 Policy State

Proposed:

\[
s_t=
[
H_t,
O_t,
E_t,
G_t,
History_t
]
\]

where:

### Human \(H_t\)

Keep using:

```text
person_trajectory [13,4]
=
[x, y, vx, vy]
```

### Object \(O_t\)

Keep using:

```text
object_trajectory [13,50,6]
=
[presence, x, y, confidence, rel_x, rel_y]
```

### Environment / Occlusion \(E_t\)

New:

```text
occlusion_trajectory [13,D_occ]
```

### Geometry \(G_t\)

Keep using:

```text
candidate angle
current angle
candidate-current delta
second harmonic
```

### History

Keep limited to:

```text
visited views
current view
remaining valid candidates
```

---

# 20. Network Change: Add a Low-Capacity Occlusion Branch

Do not switch to a large Transformer directly.

Proposed:

```text
Occlusion branch:

Conv1d(D_occ -> 16, kernel=3, padding=1)
GroupNorm
GELU
Flatten
Linear(16*13 -> 32)
LayerNorm
GELU
```

producing:

```text
occlusion context = 32-D
```

Original context:

```text
person 32-D
object 64-D
angle/current 16-D
```

New context:

```text
person 32
+ object 64
+ occlusion 32
+ angle 16
```

then project to:

```text
context 64-D
```

The candidate branch stays at the current 64-D.

Do not increase the backbone capacity first.

---

# 21. State Ablation That Must Be Done

Compare at least:

```text
A. Geometry only

B. Geometry + Person trajectory

C. Geometry + Person + Object

D. Geometry + Occlusion

E. Geometry + Person + Occlusion

F. Geometry + Person + Object + Occlusion
```

The core question is not whether a bigger network helps, but:

> does explicit environment/occlusion state genuinely produce view-selection signal across unseen classes?

---

# 22. Phase 0: Measure Active-Perception Pressure First

Before training the v2 policy, for:

```text
Original
Mild
Medium
Heavy
```

compute with fixed view0:

```text
Single
Random-2
Random-3
Preferred Orthogonal-2
Preferred Orthogonal-3
Oracle-2
All valid
Recoverable ratio
View-sensitive ratio
```

---

# 23. Gate 0: Is the Synthetic Benchmark Valid?

Continue training the policy only if most of the following conditions are met.

## Condition A

As occlusion goes:

```text
Original -> Mild -> Medium
```

single-view accuracy should degrade reasonably.

It must not happen that:

```text
heavier masks lead to overall higher accuracy
```

---

## Condition B

The Oracle-Random gap should widen markedly.

The current original data already has a noticeable Oracle gap.

Under Medium occlusion, the:

```text
Oracle-2 - Random-2
```

should widen over Original by at least roughly:

```text
+3 percentage points
```

or in relative terms:

```text
>= 30%
```

---

## Condition C

The recoverable ratio should increase markedly.

In the current original experiments, only a small fraction of samples are truly recoverable.

Suggested target:

```text
Medium occlusion recoverable >= 25%
```

More ideally:

```text
30%+
```

---

## Condition D

All views must not collapse together.

If:

```text
Oracle is also close to Random / Single
```

the mask is too heavy and active movement cannot help.

Such a benchmark is invalid.

---

# 24. Occlusion Seed / Layout Split

The same mask pattern must not repeat across train and test.

Occluder parameters are randomly sampled per episode.

Suggested:

```text
Train occlusion seeds:
1000-1999

Validation:
2000-2199

Test:
3000-3299
```

And ensure:

```text
theta_occ
width
strength
renderer
```

all vary randomly.

Additionally perform:

```text
held-out renderer
held-out occlusion strength range
```

to test generalization of the spatial strategy.

---

# 25. Policy Training Set

Still use seen classes only.

Recommended training mixture:

```text
25% original
25% mild
35% medium
15% heavy
```

Do not train everything as Heavy.

Goal:

> the policy learns when geometry/environment matter, rather than adapting only to a single fixed occlusion strength.

---

# 26. Keep VPOCLIP Frozen First

First phase:

```text
VPOCLIP frozen
```

This is necessary

because we first need to isolate:

> whether the active view policy can select better viewpoints thanks to environment state.

If VPOCLIP is fine-tuned at the same time:

```text
the recognizer may learn to adapt to the masks directly
```

which would instead remove the active-view pressure.

Only after the policy succeeds, run separately:

```text
masked-VPOCLIP robustness fine-tuning
```

as an additional experiment.

---

# 27. Utility Must Be Recomputed from Masked HAR Logits

Keep the current delta-margin utility:

\[
m(s)
=
L_{GT}(s)
-
\max_{c\neq GT}L_c(s)
\]

candidate utility:

\[
u(s,a)
=
m(s\cup a)-m(s)
\]

But all \(L\) must come from:

```text
masked / synthetic-occlusion VPOCLIP cache
```

Do not keep using the original clean logits.

---

# 28. Boundary Between Training Target and State

Allowed:

```text
future candidate masked logits
GT label
```

for offline generation of the training target:

\[
u(s,a)
\]

but forbidden from entering the policy state.

At inference, allow only:

```text
observed person trajectory
observed object trajectory
observed occlusion state
current/history geometry
candidate geometry
```

---

# 29. Keep the v1 Loss for Now

Do not change too many variables at once in the first version.

Keep:

\[
\mathcal L
=
\mathcal L_{listwise}
+
0.5\mathcal L_{pairwise}
\]

soft target:

\[
p_a
=
softmax(u_a/T_u)
\]

Initial:

```text
T_u = 0.20
```

pairwise:

```text
compare only pairs with |u_i-u_j| > 0.02
```

Stay as consistent with v1 as possible.

---

# 30. Add View-Sensitivity Weight (Ablation Recommended)

Optionally test:

\[
VS(s)
=
\max_a u(s,a)-\min_a u(s,a)
\]

sample weight:

\[
w_s
=
clip(VS(s),w_{min},w_{max})
\]

Compare:

```text
unweighted
vs
view-sensitivity weighted
```

If synthetic occlusion genuinely creates action differences, VS weighting should be more meaningful than on the original ETRI.

---

# 31. Still No PPO in the First Phase

First train the current supervised/offline ranking policy.

Rationale:

> This round first verifies whether environment state can predict useful movement.

If supervised candidate utility cannot even be learned, there is no reason to go straight to PPO.

Only after the supervised policy stably beats Random under fixed view0 / unseen settings, move on to real RL.

---

# 32. Main Evaluation Protocol: Fixed View0

Make:

```text
fixed view0
```

the main protocol.

Reason:

Previously the policy showed small positive gains from other starting views, but did not stably beat random from fixed view0.

The most important question this round is:

> can synthetic occlusion + environment-conditioned state produce stable positive gains from fixed view0 as well?

Treat:

```text
view1
view2
view3
```

as supplementary.

---

# 33. Baselines That Must Be Compared

Compare at each occlusion level:

```text
Single
Random-2
Random-3
Preferred Orthogonal-2
Preferred Orthogonal-3

Geometry-only learned scorer
Current v1 trajectory policy
v2 + occlusion branch

Oracle-2
Oracle-3
All valid
```

---

# 34. Random Must Use Multiple Seeds

Keep the current strict practice:

```text
>= 10 random evaluation seeds
```

Do not compare the policy against a single random trajectory.

Report:

```text
mean
std
paired difference
bootstrap CI
```

---

# 35. Primary Success Metric

## Primary

fixed view0, true unseen, Medium occlusion:

\[
Policy - Random
\]

Consider progress meaningful only when it reaches:

```text
>= +1.0 pp
```

and all 3 training seeds agree in direction.

More ideally:

```text
>= +2.0 pp
```

---

## Statistical

paired bootstrap:

```text
95% CI
```

Ideally:

```text
lower bound > 0
```

If the CI still crosses 0 but all 3 seeds are positive, record it as promising, not as final success.

---

# 36. A More Important Diagnostic: Action Agreement Is Not the Main Metric

Do not primarily optimize:

```text
whether the unique oracle candidate is hit
```

because multiple candidates may yield similar HAR results.

Report mainly:

```text
final Top-1
candidate regret
utility regret
Policy - Random
```

where:

\[
Regret
=
u_{oracle}-u_{selected}
\]

---

# 37. Check Whether the Policy Really Avoids Occlusion

Add the following analysis.

For each policy movement, compute:

```text
observed occlusion before move
observed occlusion after move
```

Compute:

\[
\Delta Occ
=
Occ_{before}-Occ_{after}
\]

Compare:

```text
Policy
Random
Preferred Orthogonal
```

If the policy truly learned spatial avoidance, one should see:

```text
the policy's average occlusion reduction > Random
```

---

# 38. Side-Step Behavior Analysis

Group by the position of the current mask relative to the person's left/right:

```text
mask predominantly left of person
mask predominantly right of person
mask centered
```

Examine the relative candidate angle chosen by the policy.

Ideally:

```text
occlusion on the left -> the policy prefers moving to the side that resolves the occlusion
occlusion on the right -> the opposite direction
```

This demonstrates more than Top-1 alone:

> the policy learned an environment-conditioned movement strategy.

---

# 39. Counterfactual Test

This is a very important experiment for this round.

Keep the same:

```text
human trajectory
object trajectory
camera geometry
```

unchanged; change only:

```text
occluder bearing
```

For example:

```text
theta_occ = -45°
theta_occ = +45°
```

Check whether the policy action changes systematically.

If the action barely changes:

> the policy still relies mainly on camera prior / trajectory shortcut.

If the action changes with the obstacle side:

> the environment branch is truly controlling the decision.

---

# 40. Occlusion-Feature Shuffle Test

At test time, randomly shuffle across samples:

```text
occlusion descriptors
```

Keep all other state unchanged.

If policy performance drops markedly, it does use environment state.

If it barely changes:

> the occlusion branch is ignored.

---

# 41. Geometry Shuffle Test

Likewise, shuffle candidate geometry.

If performance does not drop:

> the network may be memorizing camera slots.

Check for and block the camera-ID shortcut.

---

# 42. Camera Slot Invariance

Because view0/1/2/3 currently have no fixed left/front/right/back semantics per recording, keep ensuring this round:

```text
do not feed slot ID as bearing input
```

Optionally evaluate with randomly permuted view slots.

As long as geometry and data are permuted together correctly, policy outputs should remain geometrically equivalent.

---

# 43. Seen-to-Unseen Generalization

Train only on seen action classes.

true unseen:

```text
[A001, A003, A027, A035, A051]
class id [0,2,26,34,50]
```

Keep the final 5-way protocol.

But mask parameters must also be unseen:

```text
test occlusion seeds must not appear in training
```

The final evaluation is therefore a double generalization:

\[
\boxed{
\text{unseen action}
+
\text{unseen occlusion layout}
}
\]

---

# 44. If v2 Succeeds, Then Move to Real RL

Only after:

```text
fixed view0
true unseen
Medium occlusion
Policy > Random by meaningful margin
```

enter PPO / sequential RL.

---

# 45. State for the RL Stage

Keep using:

```text
person trajectory
object trajectory
observed occlusion trajectory
current/history geometry
visited views
```

May add:

```text
current VPOCLIP uncertainty
```

But do not input the action class.

---

# 46. RL Actions

In the 4-view dataset:

```text
STOP
move to unvisited view1
move to unvisited view2
move to unvisited view3
```

Illegal / already-visited actions are masked out.

---

# 47. RL Reward

In the synthetic occlusion environment, visibility finally has causal meaning.

Proposed:

\[
r_t
=
\alpha \Delta HAR_t
+
\beta \Delta Occ_t
+
\gamma \Delta InteractionVisibility_t
-
\lambda C_{move}
-
\mu C_{view}
\]

where:

```text
Delta HAR:
recognition evidence improvement

Delta Occ:
true observed occlusion reduction

Delta InteractionVisibility:
recovery of hand/object/upper-body key regions

C_move:
movement angle / distance cost

C_view:
fixed cost per additional observed view
```

---

# 48. RL: Do Not Let Visibility Dominate

Although synthetic occlusion makes visibility meaningful, the final task is still HAR.

Therefore:

```text
Delta HAR / final HAR reward
```

must be the main task.

Visibility is only a shaped reward.

Avoid a policy that only pursues:

```text
"seeing the most body"
```

without improving recognition.

---

# 49. If Supervised v2 Still Fails

If the benchmark already passes Gate 0:

```text
Oracle-Random gap widens markedly
recoverable ratio increases markedly
```

but v2 still has:

```text
Policy ~= Random
```

then diagnose further:

### Case A

occlusion state shuffle does not affect the policy:

```text
the environment branch learned nothing
```

Change the representation / optimization.

### Case B

occlusion state predicts oracle utility, but the network fails to learn:

```text
a capacity / training issue
```

Only then consider a larger network.

### Case C

oracle utility still correlates nearly 0 with observed environment state:

```text
the current state is still insufficient
```

Do not keep stacking networks at this point.

---

# 50. When a Larger Network Is Allowed

Only when at least:

```text
environment features correlate noticeably with utility
the small v2 model clearly beats the geometry baseline on seen/validation
a stable positive gap already appears on true unseen
```

may you test:

```text
context 128-D
person 64-D
object 128-D
occlusion 64-D
candidate 128-D
```

Then consider:

```text
GRU / TCN
small Transformer
```

Not as the first step now.

---

# 51. Optional Phase: Calibration-Based 3D Occluder

If camera calibration becomes available later:

```text
intrinsics
extrinsics
```

upgrade to a true 3D virtual cuboid.

Place in human/world coordinates:

```text
3D cuboid / vertical plane
```

Project into each camera respectively:

\[
M_v=\Pi(P_v,O)
\]

This makes synthetic occlusion closer to a real robot observing around occluders.

But this is not a prerequisite for this round's MVP.

---

# 52. Final Recommended Execution Order

```text
STEP 0
Reproduce the current v1 baseline

STEP 1
Implement the view-consistent synthetic occlusion generator

STEP 2
Generate Original / Mild / Medium / Heavy masked caches

STEP 3
Re-run frozen VPOCLIP logits

STEP 4
Evaluate Oracle-Random gap / recoverable ratio
-> Gate 0

STEP 5
Add the observed occlusion/environment branch

STEP 6
Train the supervised v2 ranker

STEP 7
Run the state ablation

STEP 8
fixed view0 true-unseen evaluation

STEP 9
Run occlusion shuffle / counterfactual / geometry shuffle

STEP 10
Only if v2 stably beats Random
move on to sequential PPO
```

---

# 53. Direct Execution Task for Codex

```text
Based on:

rl/multistep_angle_object_trajectory_policy_v1.py

create a new version:

rl/multistep_environment_occlusion_policy_v2.py

and build the synthetic occlusion data generation and evaluation scripts.

Do not train PPO directly.

==================================================
A. SYNTHETIC OCCLUSION GENERATOR
==================================================

Add:

rl/build_view_consistent_occlusion_cache.py

Requirements:

1. Use the real view_geometry sin/cos of each sample;
2. Sample one human-relative occluder bearing per episode;
3. The 4 views of the same episode share the same occluder;
4. Mask strength varies with the camera-occluder angular difference;
5. Temporally consistent across the 13 frames;
6. The mask prioritizes covering the person / upper-body / hand-object region;
7. The mask must not be decided by the GT action class;
8. At least one view stays relatively clear;
9. Save occlusion metadata;
10. Use different occlusion seeds for train/val/test.

Implement:
- Mild
- Medium
- Heavy

==================================================
B. MASKED VPOCLIP CACHE
==================================================

Use the current frozen VPOCLIP.

Recompute for masked clips:

- per-view logits
- fused evaluation cache

If compute allows:
re-run the pose/object detector.

If not for now:
keep clean pose/object,
but explicitly record that this is the clean-track synthetic setting.

==================================================
C. PHASE-0 DIAGNOSTIC
==================================================

Before training the new policy, for:

Original
Mild
Medium
Heavy

compute with fixed view0:

- Single
- Random-2
- Random-3
- Preferred Orthogonal-2
- Preferred Orthogonal-3
- Oracle-2
- Oracle-3
- All valid
- recoverable ratio
- view-sensitive ratio
- Oracle-Random gap

If Medium does not markedly increase the
Oracle-Random gap and the recoverable ratio,

stop; do not train a new policy.
Fix the synthetic occlusion first.

==================================================
D. OCCLUSION STATE
==================================================

For observed views, construct:

occlusion_trajectory [13,D]

Include at least:

- mask area ratio
- person overlap
- torso overlap
- upper-body overlap
- left-hand overlap
- right-hand overlap
- interaction overlap
- mask center relative x/y
- relative width/height

Strictly forbid inputting the true occlusion of future candidates.

==================================================
E. POLICY V2
==================================================

Keep from v1:

- person branch
- object branch
- candidate geometry
- history
- action mask

Add a low-capacity occlusion branch:

Conv1d(D_occ -> 16)
GroupNorm
GELU
Flatten
Linear -> 32
LayerNorm
GELU

context:
person 32
object 64
occlusion 32
angle 16
-> context 64

Do not enlarge the candidate branch for now.

==================================================
F. TRAINING TARGET
==================================================

Recompute using masked VPOCLIP logits:

m(s)
u(s,a) = m(s+a) - m(s)

Keep:

listwise
+
0.5 * pairwise

pair:
|u_i-u_j| > 0.02

Keep the main v1 hyperparameters in the first version;
do not change too many variables at once.

==================================================
G. TRAIN MIX
==================================================

Suggested:

25% original
25% mild
35% medium
15% heavy

Train only on seen action classes.

No hyperparameter tuning on true unseen.

==================================================
H. STATE ABLATION
==================================================

Must compare:

A Geometry
B Geometry + Person
C Geometry + Person + Object
D Geometry + Occlusion
E Geometry + Person + Occlusion
F Geometry + Person + Object + Occlusion

==================================================
I. MAIN EVALUATION
==================================================

Main protocol:

fixed view0
true unseen 5-way

Additionally evaluate fixed1/2/3.

Use >=10 evaluation seeds for Random.

Use >=3 training seeds for the policy.

Compare:

Single
Random
Preferred Orthogonal
v1 trajectory policy
v2 environment-conditioned policy
Oracle
All valid

Report:

Top-1
Policy-Random
paired bootstrap 95% CI
utility regret
movement cost

==================================================
J. BEHAVIOR DIAGNOSTICS
==================================================

Must add:

1. occlusion reduction after move
2. action histogram grouped by mask-left / mask-right
3. counterfactual occluder-bearing flip
4. occlusion-feature shuffle
5. geometry shuffle
6. camera-slot permutation test

The goal is not just accuracy,
but to prove the policy truly uses environment state.

==================================================
K. SUCCESS CONDITION
==================================================

Main goal:

fixed view0
true unseen
Medium occlusion

Hope to achieve:

Policy - Random >= +1.0 pp

More ideally:
>= +2.0 pp

And:

all 3 training seeds agree in direction.

If the paired bootstrap 95% CI lower bound > 0,
the result can be considered stable.

==================================================
L. DO NOT TRAIN PPO YET
==================================================

Start sequential PPO only after supervised v2
stably beats Random under the main protocol above.

If v2 is still roughly equal to Random,
first analyze:
- environment utility correlation
- occlusion shuffle
- counterfactual behavior

Do not automatically enlarge the network.

==================================================
FINAL REPORT
==================================================

The final report must answer:

1. Does synthetic occlusion enlarge the Oracle-Random gap?
2. Does the recoverable ratio increase markedly?
3. Does explicit occlusion state improve unseen view selection?
4. Does the policy truly learn to avoid occlusion, rather than memorizing camera priors?
5. Does fixed view0 finally beat Random stably?
6. Do the gains come from environment conditioning, or from geometry alone?
7. Is there enough evidence to proceed to PPO?
```

---

# 54. Most Important Paper Hypothesis for This Round

Suggested formulation:

> **The failure of learned active-view policies on the original ETRI setting may be caused not only by weak policy models, but by insufficient active-perception pressure: most actions are already well observed from multiple cameras.**

Then verify:

> **When viewpoint-dependent occlusion makes observability depend on camera motion, a class-agnostic policy conditioned on human–environment spatial dynamics should learn transferable viewpoint behavior without using action semantics.**

---

# 55. If This Round Succeeds, the Method Can Ultimately Be Described As

\[
\boxed{
\textbf{Environment-Conditioned Class-Agnostic Active View Acquisition}
}
\]

The policy learns:

\[
Q_\theta(
\text{human dynamics},
\text{environment/occlusion},
\text{view geometry},
\text{history},
a
)
\]

rather than:

\[
Q_\theta(
\text{action identity},
a
)
\]

This is the core transfer mechanism this project hopes for:

\[
\boxed{
\text{transfer spatial observation strategy across unseen actions}
}
\]
