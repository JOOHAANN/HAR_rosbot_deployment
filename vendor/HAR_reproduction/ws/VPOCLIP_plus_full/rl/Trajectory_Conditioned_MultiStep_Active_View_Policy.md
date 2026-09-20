# Trajectory-Conditioned Multi-Step Active-View Policy

Version: multistep_angle_object_trajectory_policy_v1

This document describes the approach that is currently implemented and tested: using sample-level camera azimuth, human trajectory, and object-detection tracks to select the next unobserved view step by step, and fusing the observed views for action recognition.

The term "policy" here follows the project's RL / active-view naming, but the current implementation is neither end-to-end PPO nor a classic DQN in an online environment. It is a low-capacity candidate-view scoring network trained with ranking targets built from offline recognition utility on seen classes; VPOCLIP is always frozen. The purpose is to first isolate the question of whether sensor state can predict useful views.

## 1. Research Goal

For an action sample, the robot starts observing from one low view. At each step the policy can use only the sensor information obtained so far to select the next view from the valid candidate views that have not yet been observed:

\[
a_t=\arg\max_{a\in\mathcal A_t}Q_\theta(s_t,a)
\]

where:

- a_t is the view-slot index, not an action class and not a continuous angle;
- s_t is the causal state formed by the currently and previously observed views;
- A_t contains only valid, reachable, and not-yet-visited views;
- VPOCLIP handles action recognition; the policy handles only view selection.

Current data has at most 4 valid low views per sample, so the meaningful upper bound on moves is 3:

    0 moves: fuse 1 view
    1 move: fuse 2 views
    2 moves: fuse 3 views
    3 moves: fuse 4 views

More than 3 moves only revisits already-seen views. Repeating a view in the offline cache produces no new observation, so it cannot count as an experiment with new visual information.

## 2. Data and Splits

### 2.1 Input Videos and Temporal Sampling

- Reference resolution of the raw videos: 640 × 480;
- Each view uses the same 13 temporal frames as the VPOCLIP cache;
- object detection runs frame by frame on these 13 frames;
- The track cache does not define a new frame-sampling rule; it reuses the sample_frames of the VPOCLIP cache.

The per-sample shape of the object track cache is:

    [4 views, 13 frames, 50 object classes, 4 channels]

The 4 raw object channels are:

    [presence, x, y, confidence]

x/y are normalized image coordinates in [-1, 1], with the y axis pointing downward.

### 2.2 Class Splits

Training uses only seen classes. The validation set uses pre-specified pseudo-unseen seen classes for model selection; real unseen classes are used only in the final test and must not participate in hyperparameter tuning.

Real unseen uses a strict 5-way evaluation:

    [A001, A003, A027, A035, A051]
    class id = [0, 2, 26, 34, 50]

At evaluation time, predictions are first restricted to these 5 classes; it is not a general prediction over all 55 classes.

### 2.3 View Azimuth Mapping

View slots 0/1/2/3 have no fixed "left/front/right/back" semantics. Each recording uses its own camera azimuth information:

    view_geometry[..., 0:2] = [sin(relative_angle), cos(relative_angle)]

Therefore, the policy uses the per-sample, per-view relative azimuth rather than treating the camera ID as an azimuth label. If view3 of one recording is on the person's right, view3 of another recording may be on the left; the network distinguishes them through the geometry vectors.

Sample-level geometry is also used to compute the relative angle between candidates:

\[
\sin(\Delta\theta_v)=
\sin\theta_v\cos\theta_c-\cos\theta_v\sin\theta_c
\]

\[
\cos(\Delta\theta_v)=
\sin\theta_v\sin\theta_c+\cos\theta_v\cos\theta_c
\]

where c is the current view and v is the candidate view.

## 3. Policy State

The state strictly follows the causal constraint: the policy can only see what has been observed in the current and past views, and cannot read future-view RGB, VPOCLIP features, or future occlusion information.

### 3.1 Human Trajectory

The cached pose is 13 × 17 × 2 = 442 dimensions. Each frame takes four torso points from COCO17:

    left/right shoulders: 5, 6
    left/right hips: 11, 12

Compute the torso center and velocity to form:

    person_trajectory: [13, 4]
    [center_x, center_y, velocity_x, velocity_y]

If multiple views have been observed, the current implementation averages the trajectories of the observed views to form a historical sensor summary. No future-view pose is used.

### 3.2 Object Tracks

The raw object track is:

    [13, 50, 4]
    [presence, x, y, confidence]

The object's position relative to the human torso center is further added:

    relative_x = object_x - person_center_x
    relative_y = object_y - person_center_y

The final object state fed to the policy is:

    object_trajectory: [13, 50, 6]
    [presence, x, y, confidence, relative_x, relative_y]

This lets the policy distinguish:

- whether an object appears;
- the object's absolute position in the image;
- the object's position relative to the human body;
- whether the object position changes over time;
- whether the detection confidence is stable.

### 3.3 Candidate-View Geometry State

Geometry is constructed for all 4 candidate views at each step. The basic inputs are:

    candidate view geometry       [sin(theta_v), cos(theta_v)]
    candidate-current delta       [sin(delta), cos(delta)]
    current view geometry         [sin(theta_c), cos(theta_c)]

Second-order harmonic terms are then added:

    [2*sin(theta)*cos(theta), cos(theta)^2 - sin(theta)^2]

The final candidate geometry feature is 10-dimensional; validity is not treated as an ordinary numeric feature but is handled separately through the action mask.

Candidate validity is:

    raw.valid
    AND raw.reachable[current_view]
    AND candidate_not_already_observed

The Q value of invalid candidates is set to negative infinity before argmax.

### 3.4 Content Explicitly Excluded from the State

The current trajectory policy does not take as input:

    RGB/video embedding
    VPOCLIP 512-D z
    skeleton embedding
    semantic belief
    current VPOCLIP logits
    action class ID
    features, logits, or object info from future views
    GT action label

VPOCLIP logits are used only to construct offline utility labels during training and to compute HAR in the final evaluation; they do not enter the policy input.

## 4. Network Architecture

The policy network is a low-capacity, factorized candidate scorer.

### 4.1 Human Branch

    Conv1d(4 -> 16, kernel=3, padding=1)
    GroupNorm
    GELU
    Flatten
    Linear(16*13 -> 32)
    LayerNorm
    GELU

It outputs a 32-D human context vector.

### 4.2 Object Branch

    50*6 = 300 input channels
    Conv1d(300 -> 64, kernel=3, padding=1)
    GroupNorm
    GELU
    Flatten
    Linear(64*13 -> 64)
    LayerNorm
    GELU

It outputs a 64-D object context vector.

### 4.3 Geometry and Candidate Branch

    current angle [2] -> Linear -> 16-D
    person 32-D + object 64-D + angle 16-D -> context 64-D
    candidate geometry [10] -> candidate MLP -> 64-D

The action score is a low-capacity matching score between the context and candidate vectors, plus a small residual branch:

\[
Q(s,a)=
\frac{h(s)^T g(a)}{\sqrt{64}}
+0.20\,r(h(s),g(a))
\]

The network finally outputs 4 candidate-view scores:

    [Q(view0), Q(view1), Q(view2), Q(view3)]

It outputs neither action classes nor continuous coordinates.

## 5. Multi-Step Decision Process

### 5.1 One Move

Given start v0:

    s1 = state(history=[v0])
    a1 = argmax Q(s1, a)
    path = [v0, a1]
    final_logits = mean(logits[v0], logits[a1])

Fuses 2 views.

### 5.2 Two Moves

After a1 is selected in the first step, the robot obtains the sensor information of the second view:

    s2 = state(history=[v0, a1])
    a2 = argmax Q(s2, a)
    path = [v0, a1, a2]
    final_logits = mean(logits[v0], logits[a1], logits[a2])

The second step does not predict anew from a single view; it uses the history state formed by the first two observed views.

### 5.3 Three Moves

    s3 = state(history=[v0, a1, a2])
    a3 = argmax Q(s3, a)
    path = [v0, a1, a2, a3]
    final_logits = mean(logits[v0], logits[a1], logits[a2], logits[a3])

If all four views are valid, every legal path ends up observing the same set. Therefore the four-view mean logits are independent of the visit order, and the policy can no longer change classification information; it can only affect the move order and the movement cost.

## 6. Training Objective

### 6.1 Seen Training Samples

Current training uses complete four-view seen episodes:

    training episodes = 2734
    training class bank = seen classes

Only complete four-view samples are used for training; samples with missing views do not enter the main training pool. The evaluation stage keeps the real valid-view mask.

The history stage is randomly generated per batch:

    history_count = 1: 45%
    history_count = 2: 45%
    history_count = 3: 10%

Therefore the network mainly learns "selecting the second view after one observed view" and "selecting the third view after two observed views"; the third move / fourth-view prediction accounts for only a small fraction.

### 6.2 Offline Delta-Margin Utility

For the seen-class action bank, define the GT margin of the current fused state:

\[
m(s)=L_{GT}(s)-\max_{c\ne GT}L_c(s)
\]

For a candidate view a, add its VPOCLIP logits to the current history fusion and construct:

\[
u(s,a)=m(s\cup a)-m(s)
\]

u(s,a) denotes the candidate's offline improvement of the seen-class recognition margin.

Important causal boundary:

- u(s,a) uses future-candidate logits to generate the training target;
- future-candidate logits do not enter the state;
- at inference time the policy has only human, object, and geometry sensors;
- the GT label does not enter the state.

### 6.3 Listwise + Pairwise Ranking Loss

First convert the candidate utilities into a soft target distribution:

\[
p_a=\operatorname{softmax}(u_a/0.20)
\]

Then use listwise cross-entropy so that candidates with higher utility receive higher scores.

Meanwhile, add pairwise ranking constraints for candidates with clearly different utilities:

    only compare candidate pairs with |u_i - u_j| > 0.02
    the order of Q_i and Q_j must match the utility order
    the larger the utility gap, the larger the required Q separation

The total loss is:

\[
\mathcal L=
\mathcal L_{listwise}+0.5\mathcal L_{pairwise}
\]

The history_count=3 stage uses a smaller weight of 0.10, because the current main targets are the first two view selections.

### 6.4 Current Training Hyperparameters

    epochs              = 40
    updates_per_epoch   = 64
    batch_size          = 16384
    learning_rate       = 1e-4
    optimizer           = AdamW
    weight_decay        = 0.04
    scheduler            = CosineAnnealingLR
    training seeds      = 20260909, 20260910, 20260911
    movement cost       = not included in training loss

movement cost is only recorded during evaluation; v1 does not use movement cost to influence action selection.

## 7. Evaluation Protocol

### 7.1 Start Views

Fix each of the following:

    fixed0
    fixed1
    fixed2
    fixed3

A random start may additionally be reported separately, but random-start and fixed-view0 results must not be mixed into the same fair protocol.

### 7.2 Compared Methods

Compare for each start view and move count:

    Single       uses only the start view
    Random       randomly selects the same number of unvisited views
    Policy       the current trajectory policy selects views step by step
    Oracle       uses the GT class to select the path with the best final margin; upper bound only
    All valid    mean over all valid views; full-view reference only

Random evaluation needs multiple evaluation seeds to avoid accidental favor or disfavor from a particular random-path sample.

### 7.3 Fusion Method

The current fusion does not concatenate or reproject 512-D embeddings; it directly averages the class logits of frozen VPOCLIP:

\[
L_{fused}=
\frac{1}{|H|}\sum_{v\in H}L_v
\]

The final prediction is the class with the highest score within the corresponding class bank.

## 8. Current Experimental Results

The following are results on real unseen 5-way, 414 samples, and 3 training seeds, repeatedly tested with 10 random evaluation seeds. In the tables, Policy is the mean over 3 training checkpoints and Random is the mean over 10 random-path seeds.

### 8.1 One Move: Fusing 2 Views

| Start | Single | Random | Policy | Policy - Random |
|---|---:|---:|---:|---:|
| view0 | 49.28% | 52.46 ± 0.49% | 52.33 ± 0.46% | -0.13 |
| view1 | 49.52% | 50.87 ± 0.63% | 51.29 ± 0.11% | +0.42 |
| view2 | 48.79% | 51.04 ± 0.77% | 52.09 ± 0.30% | +1.05 |
| view3 | 50.24% | 53.12 ± 0.70% | 53.70 ± 0.41% | +0.59 |

Averaged over the four start views: Policy 52.36%, Random 51.87%, an improvement of about +0.48 percentage points.

### 8.2 Two Moves: Fusing 3 Views

| Start | Single | Random | Policy | Policy - Random |
|---|---:|---:|---:|---:|
| view0 | 49.28% | 53.74 ± 0.53% | 53.30 ± 0.30% | -0.44 |
| view1 | 49.52% | 52.61 ± 0.72% | 53.62 ± 0.20% | +1.01 |
| view2 | 48.79% | 52.46 ± 0.62% | 52.74 ± 0.50% | +0.27 |
| view3 | 50.24% | 53.57 ± 0.50% | 53.78 ± 0.89% | +0.21 |

Averaged over the four start views: Policy 53.36%, Random 53.10%, an improvement of about +0.26 percentage points.

### 8.3 Three Moves: Fusing 4 Views

When all 4 views are valid, the final classification accuracy of Random, Policy, and All-valid is all:

    53.14%

The reason is not policy failure: after three moves all four views have been fused, and the visit order cannot change the mean logits. In the current tests, Policy mainly shows up as a slightly lower movement cost than random paths.

## 9. Interpretation of Results

The current results support the following cautious conclusions:

1. Human/object tracks and sample-level camera geometry contain a certain transferable view-selection signal;
2. This signal is more evident with view1–view3 start views;
3. Fixed view0 is the strictest and fairest main protocol, but the current policy does not consistently beat random under this protocol;
4. After repeated random evaluation, the accidental advantage of a single seed disappears; a single random path must not be reported alone;
5. Two moves usually achieve higher classification accuracy than one move, but the third view may also introduce conflicting information, so the gain is not monotonic;
6. With four-view data, three moves are mainly a path-planning problem rather than a view-selection recognition problem.

The current policy is more like "weak view ranking using observable sensor priors" and not yet a general policy that can reliably predict the best HAR view.

## 10. Three Concepts That Must Not Be Confused

### 10.1 The Policy Is Not a Class Classifier

The policy outputs:

    view index 0/1/2/3

VPOCLIP outputs:

    logits for 55 action classes

In unseen testing, the VPOCLIP logits are further restricted to the 5 real unseen classes.

### 10.2 The Training Target Is Not the Policy State

Future-view logits can be used offline to compute the target of "whether this view helps seen-class recognition", but they cannot serve as policy input. Otherwise future information leaks, and what is measured is a privileged selector rather than a deployable policy.

### 10.3 Three Moves Do Not Mean Three Rounds of New Candidate-Selection Gains

After three moves over the four fixed views, all paths observe the same four data blocks. Therefore:

- The classification result is determined by the four-view set;
- The visit order only affects motion cost and arrival time;
- To study whether new information still exists beyond the fourth view, intermediate camera poses, live video, or new sensor observations must be added.

## 11. Current Code and Result Locations

Main training policy:

    /home/youhan/ws/VPOCLIP_plus_full/rl/multistep_angle_object_trajectory_policy_v1.py

Object track extraction:

    /home/youhan/ws/VPOCLIP_plus_full/rl/build_object_position_tracks.py

Multi-seed evaluation:

    /home/youhan/ws/VPOCLIP_plus_full/rl/evaluate_trajectory_policy_seed_sweep.py
    /home/youhan/ws/VPOCLIP_plus_full/rl/evaluate_one_two_move_random_seed_sweep.py

One-move / two-move evaluation:

    /home/youhan/ws/VPOCLIP_plus_full/rl/evaluate_trajectory_policy_one_two_moves.py

Three-move, four-view fusion evaluation:

    /home/youhan/ws/VPOCLIP_plus_full/rl/evaluate_trajectory_policy_four_views.py

Training checkpoints:

    /home/youhan/ws/VPOCLIP_plus_full/work_dir/multistep_angle_object_trajectory_policy_v1/seed_20260909/best.pt
    /home/youhan/ws/VPOCLIP_plus_full/work_dir/multistep_angle_object_trajectory_policy_v1/seed_20260910/best.pt
    /home/youhan/ws/VPOCLIP_plus_full/work_dir/multistep_angle_object_trajectory_policy_v1/seed_20260911/best.pt

Current main results:

    /home/youhan/ws/VPOCLIP_plus_full/work_dir/multistep_angle_object_trajectory_policy_v1/summary.json
    /home/youhan/ws/VPOCLIP_plus_full/work_dir/multistep_angle_object_trajectory_policy_v1/one_two_move_test/random_seed_sweep/summary.json
    /home/youhan/ws/VPOCLIP_plus_full/work_dir/multistep_angle_object_trajectory_policy_v1/four_view_three_move_test/summary.json

## 12. Possible Future Directions

If this line is continued, the priorities should be:

1. Use multi-seed validation under the fixed-view0 main protocol, rather than chasing the best number at some start view;
2. Add a STOP action so that the policy can stop moving when the second view is already good enough;
3. Report "recognition gain" and "movement cost" as independent dimensions before deciding whether to include them in the reward;
4. Use more held-out seen classes to check the predictability of view utility from tracks;
5. To study more than 3 moves, provide genuinely new observations instead of repeating the four cached views;
6. Consider real online RL algorithms such as PPO/DQN only after state predictability is clearly established.

