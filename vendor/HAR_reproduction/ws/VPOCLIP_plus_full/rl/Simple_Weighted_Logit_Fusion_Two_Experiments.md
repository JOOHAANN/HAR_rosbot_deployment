# Simple Weighted Logit Fusion: Two Lightweight Experiments

This round runs only two simple fusion experiments; no further complex models.

Common fusion form:

\[
z_{fused}=\sum_{v=1}^{K} w_v z_v,\qquad \sum_v w_v=1
\]

where \(z_v\) is the classification logits of each view.

---

## 0. Very Important: The Two Experiments Are Fully Separate

Create two separate directories:

```text
work_dir/fusion_lightweight_gating_v1/
work_dir/fusion_rule_based_v1/
```

If new scripts are needed:

```text
rl/fusion_lightweight_gating_v1.py
rl/fusion_rule_based_v1.py
```

Logs are also separate:

```text
logs/fusion_lightweight_gating_v1.log
logs/fusion_rule_based_v1.log
```

Requirements:

```text
Do not overwrite any previous code
Do not overwrite old checkpoints
Do not overwrite old caches
Do not overwrite old logs
Do not overwrite old work_dirs
Do not modify the old RL
Do not retrain VPOCLIP
```

Both experiments only read existing checkpoints / caches and write outputs to their own new directories.

---

# 1. Common Baseline

The current Equal Mean Fusion:

\[
w_v=rac{1}{K}
\]

\[
z_{mean}=rac{1}{K}\sum_v z_v
\]

All new methods must be compared against this baseline on exactly the same view sets.

Test view sets:

```text
Random-2
Preferred-Orthogonal-2
Random-3
Preferred-Orthogonal-3
All-valid views
```

---

# 2. Experiment A: Lightweight Gating Network

Directory:

```text
work_dir/fusion_lightweight_gating_v1/
```

Goal:

Use a very small shared network that, based on the quality and geometry information of each observed view, outputs a scalar weight.

---

## 2.1 Per-View Inputs

### Classification

```text
entropy              1
margin               1
max probability      1
```

Suggested:

```text
entropy = normalized entropy
margin = top1 - top2 probability margin
max probability = max softmax probability
```

### Temporal stability

Use only fields already present in the existing cache, preferring:

```text
entropy_std
top1_switch_rate
logit_variance_mean
margin_std
```

Using 2–4 in total is sufficient.

Do not fabricate fields that do not exist.

### Geometry

```text
sin(azimuth)
cos(azimuth)

sin(elevation)
cos(elevation)
```

4 dimensions in total.

### Relative diversity

The mean angular distance of each view to the other observed views:

\[
D_v=rac{1}{K-1}\sum_{j
eq v} d_{angle}(v,j)
\]

Normalize to 0–1.

### Cross-view JS

The mean JS divergence between the prediction distribution of each view and the other views:

\[
JS_v=rac{1}{K-1}\sum_{j
eq v}JS(p_v,p_j)
\]

---

## 2.2 Final Input Dimension

Approximately:

```text
entropy                 1
margin                  1
max probability         1
temporal stability      2~4
azimuth sin/cos         2
elevation sin/cos       2
relative diversity      1
cross-view JS           1
--------------------------
total                 ~11-13
```

---

## 2.3 Network

Keep it very lightweight:

```text
Input D
  ↓
Linear(D, 16)
ReLU
  ↓
Linear(16, 1)
  ↓
score e_v
```

All views share the same network.

Do not use:

```text
Transformer
large MLP
512-D feature input
full logits input
camera-specific network
```

---

## 2.4 Weights and Fusion

For all observed views of a sample:

\[
w_v=softmax(e_v)
\]

Finally:

\[
z_{fused}=\sum_v w_v z_v
\]

VPOCLIP is fully frozen; only this small gating network is trained.

Loss:

\[
L=CE(z_{fused},y_{GT})
\]

The last layer is initialized near 0 so that initially:

\[
w_vpprox 1/K
\]

That is, it starts learning from the current equal mean.

---

# 3. Experiment B: Rule-Based Weighted Fusion

Directory:

```text
work_dir/fusion_rule_based_v1/
```

This experiment:

```text
does not train any network
```

Uses only:

```text
entropy
margin
skeleton visibility
body-camera orientation
```

to compute the view weight.

---

## 3.1 Entropy Score

Normalized entropy:

\[
H_v^{norm}\in[0,1]
\]

Definition:

\[
S_H(v)=1-H_v^{norm}
\]

That is, the lower the entropy, the higher the score.

---

## 3.2 Margin Score

Use the probability top1–top2 margin:

\[
M_v=p_{(1)}-p_{(2)}
\]

Normalize to 0–1:

\[
S_M(v)=M_v^{norm}
\]

---

## 3.3 Skeleton Visibility

If the current cache has a valid / visible joint mask:

\[
S_{vis}(v)=rac{visible\ joints}{total\ joints}
\]

Range 0–1.

Do not retrain the pose model.

---

## 3.4 Body Orientation

Define the angle:

```text
0°   = person facing the camera
90°  = person side-facing the camera
180° = person fully back to the camera
```

Desired behavior:

```text
30°–90° is best
facing away from the camera incurs a clear penalty
```

Use a simple rule:

```text
if angle < 30:
    score = 0.8 + 0.2 * angle / 30

elif angle <= 90:
    score = 1.0

elif angle <= 150:
    score = 1.0 - 0.6 * (angle - 90) / 60

else:
    score = 0.2
```

That is:

```text
0°      -> 0.8
30~90°  -> 1.0
150°    -> 0.4
180°    -> 0.2
```

---

# 4. Three Simple Rule-Based Versions

## Rule-1: Classification Only

\[
S_v=0.5S_H+0.5S_M
\]

That is:

```text
entropy 50%
margin  50%
```

---

## Rule-2: Skeleton / Orientation Only

\[
S_v=0.5S_{vis}+0.5S_{ori}
\]

That is:

```text
visibility  50%
orientation 50%
```

---

## Rule-3: Combined

\[
S_v=
0.30S_H
+
0.30S_M
+
0.20S_{vis}
+
0.20S_{ori}
\]

That is:

```text
entropy      30%
margin       30%
visibility   20%
orientation  20%
```

This is the main rule-based method.

---

## 4.1 Rule Weight

Finally:

\[
w_v=rac{S_v}{\sum_j S_j}
\]

Then:

\[
z_{fused}=\sum_v w_vz_v
\]

Modifying these coefficients on true unseen is forbidden.

If changes are needed, they must be based on seen validation only.

---

# 5. Evaluation

Compare:

```text
Equal Mean Fusion
Lightweight Gating
Rule-1 Classification
Rule-2 Skeleton/Orientation
Rule-3 Combined
```

Use exactly the same:

```text
Random-2
Orthogonal-2
Random-3
Orthogonal-3
All-valid
```

Report separately:

```text
Seen
True unseen
```

---

# 6. Output Table

| View Set | Mean | Lightweight Net | Rule-Cls | Rule-Geo | Rule-Combined |
|---|---:|---:|---:|---:|---:|
| Random-2 | | | | | |
| Orthogonal-2 | | | | | |
| Random-3 | | | | | |
| Orthogonal-3 | | | | | |
| All-valid | | | | | |

Each method reports:

```text
Top-1
Delta vs Equal Mean
paired bootstrap 95% CI
```

Also save:

```text
sample_id
view_ids
entropy per view
margin per view
visibility per view
orientation per view
final weights
final prediction
```

---

# 7. Direct Execution Instructions for Codex

```text
This round runs only two new fusion experiments.

Very important:
The two experiments must be placed in two completely new, separate directories.

Create:

work_dir/fusion_lightweight_gating_v1/
work_dir/fusion_rule_based_v1/

If scripts are needed:

rl/fusion_lightweight_gating_v1.py
rl/fusion_rule_based_v1.py

Logs:

logs/fusion_lightweight_gating_v1.log
logs/fusion_rule_based_v1.log

Do not overwrite or modify any old experiment.

==================================================
EXPERIMENT A
LIGHTWEIGHT GATING NETWORK
==================================================

Per-view inputs:

- normalized entropy
- top1-top2 probability margin
- max probability

temporal stability:
select 2–4 from the existing cache:
- entropy_std
- top1_switch_rate
- logit_variance_mean
- margin_std

geometry:
- sin azimuth
- cos azimuth
- sin elevation
- cos elevation

plus:
- relative angular diversity
- mean cross-view JS divergence

Total input dimension is about 11–13.

Network:

D
-> Linear(16)
-> ReLU
-> Linear(1)

All views share the same network.

weights = softmax(view_scores)

z_fused = sum(weight_v * logits_v)

Freeze VPOCLIP.
Train only this gating network.

loss:
cross entropy of fused logits.

Initialize the last layer near 0,
so that initial weights are near uniform.

==================================================
EXPERIMENT B
RULE-BASED WEIGHTED FUSION
==================================================

No network is trained.

Per view:

1. S_H = 1 - normalized_entropy

2. S_M = normalized top1-top2 probability margin

3. S_vis = visible_joint_ratio

4. orientation score:

0 deg   = facing camera
90 deg  = side
180 deg = back to camera

if angle < 30:
    score = 0.8 + 0.2 * angle / 30
elif angle <= 90:
    score = 1.0
elif angle <= 150:
    score = 1.0 - 0.6 * (angle - 90) / 60
else:
    score = 0.2

Test:

Rule-1:
0.5 * S_H
+ 0.5 * S_M

Rule-2:
0.5 * S_vis
+ 0.5 * S_ori

Rule-3:
0.30 * S_H
+ 0.30 * S_M
+ 0.20 * S_vis
+ 0.20 * S_ori

Finally:

w_v = S_v / sum(S)

z_fused = sum(w_v * logits_v)

Using true unseen to tune the coefficients is forbidden.

==================================================
COMMON EVALUATION
==================================================

Compare:

Equal Mean
Lightweight Gating
Rule-1
Rule-2
Rule-3

view sets:

Random-2
Orthogonal-2
Random-3
Orthogonal-3
All-valid

Report separately:

Seen
True unseen

Output:

Top-1
Delta vs Mean
paired bootstrap 95% CI

Save the final weights of each sample.

==================================================
IMPORTANT
==================================================

Do not continue PPO.
Do not add a Transformer.
Do not do feature-level fusion.
Do not retrain VPOCLIP.

After this round finishes, only output results and analysis;
do not automatically continue with more complex fusion.
```

---

# 8. Goals of This Round

Experiment A answers:

> Can a lightweight gating network with a ~10-dimensional input and a single hidden layer beat equal mean?

Experiment B answers:

> Without any network, is entropy + margin + skeleton visibility + body orientation already sufficient to obtain more reasonable view weights?

If both methods are almost equal to Equal Mean, stop here for now; do not immediately move to a complex network.
