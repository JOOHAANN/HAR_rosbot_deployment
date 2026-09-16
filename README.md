# Active Zero-Shot Human Action Recognition for Mobile Robots via Lightweight Multimodal Perception and Reinforcement Learning-Based View Selection

This is the official repository for the paper:

> **Active Zero-Shot Human Action Recognition for Mobile Robots via Lightweight Multimodal Perception and Reinforcement Learning-Based View Selection**
>
> Youhan Li\*†, Jinhua Xu\*¹², Antonio Sgorbissa¹, Carmine Tommaso Recchiuto¹
>
> \* Equal contribution. † Corresponding author: youhan.li@edu.unige.it
>
> ¹ Department of Informatics, Bioengineering, Robotics, and Systems Engineering (DIBRIS), University of Genoa, Italy
> ² Department of Computer, Control, and Management Engineering, Sapienza University of Rome, Italy

## Abstract

Socially assistive robots (SARs) in domestic healthcare must reliably perceive human behaviors across individuals with diverse physical conditions and evolving habits. However, real-world deployment faces fundamental challenges: behavior patterns are inherently subject-specific, making it infeasible to collect personalized training data and train closed-set models for every individual. Concurrently, strict in-home privacy regulations preclude offloading private video streams to the cloud for continual model retraining. While open-vocabulary action recognition resolves these barriers by indexing novel activities via natural-language descriptions, existing methods remain passive and rely on fixed viewpoints. Unlike closed-set models that exploit background context biases, zero-shot alignment relies strictly on articulated kinematic motion and object interactions, making it acutely vulnerable to self-occlusion and environmental clutter.

To overcome these limitations without altering living environments with intrusive fixed-camera networks, we propose an active zero-shot human action recognition framework for dual micro mobile robots. First, VPOCLIP unifies RGB clips, skeletal pose, and object geometry into a frozen language embedding space. This lightweight recognizer requires only 27.1M parameters and 130 ms latency on an NVIDIA Jetson AGX Xavier, while providing a versatile semantic interface for downstream local LLM reasoning and semantic navigation. Second, to resolve visual ambiguities within narrow action time windows while minimizing disruption to occupants, an offline Double Dueling DQN coordinates two micro robots. Learning solely on seen actions from class-agnostic motion dynamics, body orientation, and object geometry, the policy selects complementary viewpoint pairs under a minimum-movement assignment.

Evaluated on ETRI-Activity3D under a strict 45/5/5 subject-disjoint protocol, our active policy successfully generalizes to unseen categories. Compared to single-view and heuristic baselines, the cooperative policy improves recognition accuracy, achieving critical 5%–10% gains on ambiguous, occlusion-heavy action classes in simulation, while expending only 27.3% of the movement cost incurred by random exploration. Furthermore, in real-world domestic deployments, the cooperative policy mitigates off-center visual drift, demonstrating an improved specific category output rate alongside a significant reduction in physical travel distance. This establishes an extensible reinforcement learning platform that effectively integrates physical reachability, safety constraints, and cooperative perception for domestic assistive robotics.

## Repository contents

This repository contains the runtime scripts, source code, configuration files, and documentation for the dual-ROSbot deployment. Model weights, datasets, training outputs, caches, logs, and local environment files (e.g. `.env`) are not committed; see [`.gitignore`](./.gitignore). After cloning, restore the model/data assets listed under [Weights](#weights) before running `docker compose` and the verification scripts below.

## Docker startup

Compose creates one long-running ROS2/ML container; inference processes are started inside it with `docker exec`:

```bash
cd HAR_rosbot_deployment
docker compose up -d --build

docker exec -it har-rosbot-deployment bash
source /opt/ros/jazzy/setup.bash
source /workspace/person_follow_ws/install/setup.bash
source /workspace/person_follow_ws/select_rosbot.sh rosbot_1
cd /workspace/CLIPGCN

# Verify local weights and environment before starting
bash scripts/verify_container.sh
bash scripts/run_policy_smoke.sh

# Start the VPOCLIP + four-view DDQN nodes for both ROSbots (foreground)
bash scripts/run_rosbot_stack.sh
```

The stack can also run in the background:

```bash
docker exec -d har-rosbot-deployment bash -lc \
   'source /workspace/CLIPGCN/scripts/runtime_env.sh && \
   cd /workspace/CLIPGCN && \
   exec bash scripts/run_rosbot_stack.sh'
```

The container uses the `clipgcn-rosbot:jazzy` image, an NVIDIA GPU, ROS2 Jazzy, and the host ROS network. It starts:

- One `ros_realtime.py` per robot in `HAR_VPOCLIP_ROBOT_NAMESPACES`, loading the VPOCLIP, X3D, CTR-GCN, YOLO, and RTMPose weights for real-time recognition.
- A fused display in the `rosbot_1` window: the local single-view result plus the dual-view fused result from `/har/final_action` overlaid in magenta. The fusion combines the latest VPOCLIP predictions from both robots.
- `active_view_ros.py`: loads the `full_depth19` dual-robot Pair-Dueling DDQN and the RTMPose orientation MLP, subscribes to the four views, and publishes `/har/selected_view_pair`.

## Launch scripts

Run the following entry points from a host desktop terminal. Each uses Docker; Nav2 runs only in the existing `jazzy-rosbot` container.

### 1. VPOCLIP recognition windows

```bash
cd HAR_rosbot_deployment
./scripts/start_vpoclip.sh
```

Opens two windows, `CLIPGCN rosbot_1` and `CLIPGCN rosbot_2`. Press `q` or `Esc` in a window to stop that recognition process; the DDQN view-selection node creates no window. The host must allow X11 (the script runs `xhost +si:localuser:root`). `display=false` in `vpoclip_config.csv` applies only to background launches; this entry point forces the GUI on.

Both VPOCLIP instances run RTMPose on CUDA; the dual-robot launcher warms up the two recognizers serially to avoid cold-start contention on a 6 GB GPU.

### 2. Nav2

```bash
./scripts/start_nav.sh
```

Starts localization and Nav2 for both robots in the `jazzy-rosbot` container using the `home` map. Use `--rviz` to open RViz and `--restart` to restart navigation. The map is set by `map_name` in `config/orbit_system_config.csv`.

Initial robot poses do not need to be measured by hand: `initial_robot_pose_mode=relative_human_slots` derives the starting `x/y/yaw` of rosbot_1/rosbot_2 from the human position, human heading, `ring_radius_m`, and `initial_robot_slots=0;1`, with both robots facing the human. This slot binding is only a bootstrap for the first plan; during operation each plan reads the current `map -> rosbot_i/base_link` TF and updates the slot when the robot reaches its target. If navigation is already running and the human pose changes, reload the initial poses with `--restart`; a plain start does not disturb running localization:

```bash
./scripts/start_nav.sh --dry-run
./scripts/start_nav.sh --restart --rviz
```

For compatibility with manual measurement, set `initial_robot_pose_mode=manual_csv` and edit the rows with `enabled=true` in [`config/robot_initial_poses.csv`](./config/robot_initial_poses.csv). `yaw_deg` is in degrees and is converted to radians by the script.

### 3. View-movement policy (pick one)

```bash
./scripts/start_motion.sh rl       # full_depth19 DDQN
./scripts/start_motion.sh random   # random_pair
./scripts/start_motion.sh cyclic   # cyclic_pair: 0-2, 1-3 alternation
```

These modes correspond to the RL, random, and cyclic view-selection protocols and publish to `/har/selected_view_pair`. The exported package contains only the view-selection RL weights — no base-motion weights mapping map coordinates to Nav2 goals — so this script does not publish `cmd_vel`; base navigation is handled by script 2 and Nav2.

## Robot movement / view-selection configuration

File: [`config/orbit_system_config.csv`](./config/orbit_system_config.csv)

This table is the single entry point for the robot movement system. Restart the relevant scripts after editing. Common fields:

```text
selection_mode=rl/random/cyclic       DDQN, random, or cyclic view policy
dry_run=true/false                    true: print target coordinates only; false: allow Nav2 goals
ring_radius_m=1.5                     radius of the ring centred on the current human point (m)
ring_angles_deg=270;0;90;180          slot offsets relative to the human's front; slot 1 is 0° (front)
human_position_mode=fixed             manual human map coordinates; no depth/pixel back-projection;
                                      can be updated online from RViz
human_x_m=0.0 / human_y_m=0.0         starting human x/y in map; adjust to the actual scene
human_z_m=0.0                         fixed human z; diagnostics only, no effect on 2-D navigation
human_yaw_deg=0.0                     human heading in map; ROS convention, +x is 0°
human_front_slot=1                    the slot the human's front faces
initial_robot_pose_mode=relative_human_slots
                                      derive initial robot poses from the human pose and slots;
                                      set to manual_csv for manual entry. Bootstrap only; it does
                                      not lock later robot positions
human_marker_topic=/har/orbit/human_center_marker
                                      red Marker of the current human point in RViz
human_heading_marker_topic=/har/orbit/human_heading_marker
                                      yellow arrow of the human heading in RViz
human_pose_topic=/har/orbit/human_pose
                                      topic published by the RViz online human-pose tool
initial_robot_marker_topic=/har/orbit/initial_robot_markers
                                      two virtual initial-robot arrows/labels that follow the human
initial_robot_slots=0;1               initial slots only; updated on goal arrival, not a permanent binding
bootstrap_pair=0;2                    initial target slots used before any policy output arrives
active_view_variant=full_depth19      exported DDQN architecture
active_view_policy_device=cuda        DDQN device
active_view_rtmpose_device=cuda       RTMPose device used by the view policy
fusion_continuous=true                continuously fuse the two latest VPOCLIP results
```

The full system entry point starts Nav2, RViz, Follow, dual-robot VPOCLIP, fusion, the view policy, and the ring coordinator. Follow is disabled while a robot moves and enabled while it is stationary. The default `dry_run=true` only prints the targets and trajectories (rosbot_1 first, then rosbot_2) without moving the base:

```bash
./scripts/start_orbit_system.sh
```

Dry-run does not write simulated arrivals back to the real robot slots; with `dry_run_update_slot_mapping=false` the active-view module keeps working from the robots' actual slots. If either robot camera produces no new frame for more than `active_view_input_stale_seconds` (default 3 s), view selection pauses instead of reusing the stale 13-frame history.

Stop the full system with:

```bash
./scripts/stop_har_system.sh
```

This stops HAR/VPOCLIP, Follow, fusion, view selection, localization, Nav2, and the RViz instance started by this project. It does not remove Docker containers, images, or volumes.

Command-line flags override the selection mode for one run without modifying the CSV:

```bash
./scripts/start_orbit_system.sh --selection random
./scripts/start_orbit_system.sh --selection cyclic --continuous
```

`start_motion.sh` without arguments also reads `selection_mode`; passing `rl`, `random`, or `cyclic` overrides it for that run. Map-coordinate planning and the serial Nav2 dispatch are done by `orbit_coordinator.py`: it uses `human_x_m/human_y_m` at startup, and the ring centre and heading can be updated at runtime from RViz via `/har/orbit/human_pose`. The system does not subscribe to the depth image; the view policy only selects the next ring slot.

### Adjusting the human and robot display poses online in RViz

Keep the RViz `Fixed Frame` as `map`. The toolbar has two `2D Pose Estimate` tools: the stock one sets rosbot_1 localization; the other publishes to `/har/orbit/human_pose` for the human. Click the human centre on the map and drag the arrow to set the heading; the pose is published on mouse release. The coordinator immediately updates the red human sphere, the yellow heading arrow, and the blue/green "virtual initial robot poses" arrows and labels, which follow the human position and heading.

The virtual robot markers do not overwrite the real `map -> rosbot_i/base_link` TF, and dragging the human does not move the base. The real robot model in RViz still shows the true localization; the blue/green arrows show the initial relative coordinates derived from slots 0/1 and are for checking the configuration. With `persist_rviz_human_pose=true`, the released x/y/z/yaw is also written back to `config/orbit_system_config.csv` as the next startup value. To make Nav2 adopt these derived initial poses, verify the scene is safe and run `./scripts/start_nav.sh --restart`.

The four ring targets are shown via `/har/orbit/target_poses`. When the human heading changes, the map coordinates of slots 0/1/2/3 rotate accordingly; the default `270;0;90;180` puts slot 1 in front of the human, slots 0/2 on the sides, and slot 3 behind. If RViz is already open, restart it to pick up the added tools and displays.

Only after the RViz targets and logs are confirmed correct, set `dry_run=false` in the table or explicitly run `./scripts/start_orbit_system.sh --live`. During movement Follow stays off, and rosbot_2 may start only after rosbot_1 finishes. When no robot is moving, Follow turns back on but only rotates in place: the Follow controller fixes `linear.x=0` and uses only `angular.z`. The final `cmd_vel` is still arbitrated, with Nav2 taking priority over Follow. Dual-robot VPOCLIP fusion always uses the latest results from both sides and does not wait for movement to finish. `settle_seconds=3.0` still applies to the post-arrival recognition cycle and the active-view feedback flag.

## Typical startup sequence

Verify target coordinates first with a continuous dry run:

```bash
cd HAR_rosbot_deployment
./scripts/start_orbit_system.sh --dry-run --continuous
```

After confirming RViz, targets, and logs, switch to real movement:

```bash
./scripts/stop_har_system.sh
./scripts/start_orbit_system.sh --live
```

`--live` sends Nav2 goals and drives the robots; confirm the area is safe first. The map defaults to `map_name=home` in `config/orbit_system_config.csv`.

To watch the human centre, the four ring candidates, both robots' targets, and intermediate waypoints in real time, open another terminal:

```bash
./scripts/watch_orbit_targets.sh
```

This is a read-only monitor; `Ctrl-C` exits the monitor without stopping the robot system. All coordinates are in the `map` frame; in dry-run it shows `NO NAV2 GOAL SENT`.

By default `HAR_VPOCLIP_ROBOT_NAMESPACES=rosbot_1,rosbot_2` starts an independent VPOCLIP ROS node per robot. To reduce GPU memory pressure, run a single robot:

```bash
export HAR_VPOCLIP_ROBOT_NAMESPACES=rosbot_1
docker compose up -d
```

Dual-robot CUDA models initialize serially so the two RTMPose/YOLO instances do not compete for memory at cold start; the wait limit is `HAR_VPOCLIP_START_TIMEOUT_SEC` (default 120 s).

In the all-GPU configuration, both VPOCLIP recognizers, the active-view policy, and their RTMPose instances use CUDA. If GPU memory is insufficient, temporarily set `rtmpose_device` or `active_view_rtmpose_device` back to `cpu`.

## Editable configuration tables

### 1. Action candidate table

File: [`realtime_class_config.csv`](./realtime_class_config.csv)

```text
enabled=true/false   include in VPOCLIP ranking; false removes the class from the candidate pool
split=seen/unseen    assign the class to the seen or unseen candidate side
score_scale          multiplicative score calibration, normally 1
score_bias           additive score calibration, normally 0
```

The default table corresponds to `strict45_10` in `RESULTS.md`: all 55 model classes enabled, the 10 unseen classes are `1,7,14,15,18,25,39,46,52,54`, and the final test unseen classes are `25,39,46,52,54`. Action IDs are zero-based model labels; `A01` is label `0`.

For example, to disable "reading a book", change only that row's `enabled`:

```text
A32,31,reading a book,false,seen,1,0
```

Restart VPOCLIP after editing. With `candidate_scope=all`, ranking runs over all enabled seen/unseen classes; a class with `enabled=false` is excluded.

### 2. VPOCLIP parameter table

File: [`vpoclip_config.csv`](./vpoclip_config.csv)

This table centralises the model, weight, RTMPose, temporal-window, ranking, entropy-gating, YOLO, display, and follow-output parameters of `ros_realtime.py`. The launch scripts read it automatically; relative paths resolve against the project root. Common edits:

```text
predict_every,1,...
top_k,1,...
decision_entropy_threshold,0.30,...
decision_temperature,0.05,...
rtmpose_device,cuda,...
fusion_display_enabled,true,...
fusion_display_robot,rosbot_1,...
fusion_display_topic,/har/final_action,...
fusion_display_stale_seconds,15.0,...
enable_person_follow_output,false,...
```

`start_vpoclip.sh` forces `display=true` but honours all other parameters. By default both robots run VPOCLIP/YOLO/feature models on CUDA per the YAML config, with RTMPose also on CUDA. The two recognizers initialize serially and the second waits for the first to stabilise; for a temporary override set `HAR_VPOCLIP_RTMPOSE_DEVICE=cuda`. Setting `runtime_device` to `cuda` runs everything in full-CUDA mode and overrides the separate RTMPose device setting.

`runtime_device=config` is a special value: it passes no `--runtime-device` and uses `runtime.device` from `config_runtime.yaml` directly, which allows `rtmpose_device` to be set to CPU independently.

Weight paths in the table point into `vendor/`; when changing a checkpoint, also confirm that `config`, `class_split_dir`, and the CTR-GCN/X3D architectures still match.

Default camera topics:

```text
/rosbot_1/camera/rgb/image_raw/compressed
/rosbot_2/camera/rgb/image_raw/compressed
/har/view/slot_2/compressed
/har/view/slot_3/compressed
```

If your topics differ, set them before starting:

```bash
export HAR_VIEW_TOPICS=/your/topic0,/your/topic1,/your/topic2,/your/topic3
export HAR_ROBOT_NAMESPACE=rosbot_1
docker compose up -d
docker exec -d har-rosbot-deployment bash -lc \
  'source /workspace/CLIPGCN/scripts/runtime_env.sh && \
   cd /workspace/CLIPGCN && exec bash scripts/run_rosbot_stack.sh'
```

Inspect the outputs:

```bash
docker exec har-rosbot-deployment bash -lc \
  'source /opt/ros/jazzy/setup.bash && ros2 node list && ros2 topic list'
docker exec har-rosbot-deployment bash -lc \
  'source /opt/ros/jazzy/setup.bash && ros2 topic echo /har/selected_view_pair'
tail -f logs/active_view.log logs/recognizer_rosbot_1.log logs/recognizer_rosbot_2.log
```

## Using the existing jazzy navigation container

`jazzy-rosbot` is a separate navigation container; do not start Nav2 inside the HAR container. Localize both robots on the existing `home` map and start Nav2:

```bash
docker start jazzy-rosbot
docker exec -it jazzy-rosbot bash
source /opt/ros/jazzy/setup.bash
cd /root/rosbot2-jazzy-image/host/offboard

./rosbot-offboard localize rosbot_1 home
./rosbot-offboard localize rosbot_2 home
./rosbot-offboard nav rosbot_1
./rosbot-offboard nav rosbot_2

for ns in rosbot_1 rosbot_2; do
  ros2 lifecycle get /$ns/slam_toolbox
  ros2 lifecycle get /$ns/controller_server
  ros2 action info /$ns/navigate_to_pose
done
```

Then start the velocity arbiter for both robots in the HAR container:

```bash
for ns in rosbot_1 rosbot_2; do
  docker exec -d har-rosbot-deployment bash -lc \
    "source /workspace/CLIPGCN/scripts/runtime_env.sh && \
     exec ros2 launch person_follow_demo person_follow_demo.launch.py \
     namespace:=$ns >>/workspace/CLIPGCN/logs/follow_${ns}.log 2>&1"
done
```

Finally, run the HAR Docker startup commands above. Nav2 publishes to `/<robot>/cmd_vel_nav`, and the arbiter is the sole publisher of the final `/<robot>/cmd_vel`. During verification do not run `rosbot-offboard goal`; send goals only after confirming correct localization in RViz.

## Weights

All weights live under `vendor/`:

| Module | File |
|---|---|
| VPOCLIP | `vendor/HAR_reproduction/ws/VPOCLIP_plus_full/work_dir/merged45_10_groupAB/selected_stage_b/last_model.pth` |
| X3D | `vendor/HAR_reproduction/ws/X3D_full/outputs/etri_allviews_cs45_merged_groupAB_7000/model_best.pth` |
| CTR-GCN COCO-17 | `vendor/HAR_reproduction/ws/CTR-GCN_17_full/work_dir/etri_coco17/allviews_cs45_merged_groupAB/runs-51-4029.pt` |
| YOLO custom-50 | `vendor/HAR_reproduction/ws/yolov5_full/best.pt` |
| RTMPose | `vendor/HAR_reproduction/.cache/rtmlib/hub/checkpoints/` |
| View DDQN | `vendor/HAR_reproduction/ws/VPOCLIP_plus_full/work_dir/dual_robot_depth_ddqn_strict45_5/models/full_depth19/seed_20260911/best.pt` |
| Orientation MLP | `vendor/RTMPose_orientation_MLP/work_dir/rtmpose_orientation_mlp/best.pt` |

## Person following

By default the stack only recognizes and publishes results. If the follow-control nodes from `person_follow_ws` are running, enable detection output explicitly:

```bash
export HAR_ENABLE_FOLLOW=1
docker compose up -d
```

This publishes `/<robot>/follow/person_detection`; it does not start a base controller.

## Safety boundary

The exported DDQN uses four view slots and a normalized angular cost; the export contains no site-specific TF, navigation goals, or view-slot-to-Nav2-goal calibration. The node therefore only publishes view-selection results and never sends Nav2 motion commands directly. Connect a site-specific navigation adapter behind `/har/selected_view_pair` only after confirming the on-site TF, map, and cost definitions.

Without a ROS camera, run `run_policy_smoke.sh` to verify the DDQN, orientation MLP, GPU, and weight loading; the ROS nodes wait for the real image topics.

## Results

The simulation evaluation under the independent random-start protocol — unseen accuracy, movement cost, per-class breakdowns, ablations, and the full experimental procedure — is documented in [RESULTS.md](./RESULTS.md).
