# HAR ROSbot deployment

这是一个独立的新项目目录，参考了本机的 `VPOCLIP_rosbot` ROS 实时入口，并使用 `har_export` 中的完整权重。原始仓库没有被修改。

## Repository contents

本仓库保留运行脚本、源码、配置和文档；模型权重、数据集、训练输出、缓存、日志、
本地运行环境和 `.env` 等大文件或机器相关文件不会提交。新环境从 GitHub 克隆后，
需要先恢复本地模型/数据资产，再运行 `docker compose` 和下面的验证脚本。具体忽略
规则见 [`.gitignore`](./.gitignore)。

## Docker 启动方式

这里沿用 `/home/youhan/HAR/VPOCLIP_rosbot` 的部署方式：Compose 只负责创建一个长期运行的 ROS2/ML 容器，推理进程通过 `docker exec` 在容器内启动。这样可以先完成 ROSbot 选择、Follow workspace 和 DDS 配置，再启动 HAR。

```bash
cd /home/youhan/HAR/HAR_rosbot_deployment
docker compose up -d --build

# 进入与 VPOCLIP_rosbot 相同的 ROS2 Jazzy 容器
docker exec -it har-rosbot-deployment bash
source /opt/ros/jazzy/setup.bash
source /workspace/person_follow_ws/install/setup.bash
source /workspace/person_follow_ws/select_rosbot.sh rosbot_1
cd /workspace/CLIPGCN

# 启动前验证本地权重和环境
bash scripts/verify_container.sh
bash scripts/run_policy_smoke.sh

# 启动两个 ROSbot 的 VPOCLIP + 四视角 DDQN 节点（前台运行）
bash scripts/run_rosbot_stack.sh
```

也可以让 HAR 在容器后台运行：

```bash
docker exec -d har-rosbot-deployment bash -lc \
   'source /workspace/CLIPGCN/scripts/runtime_env.sh && \
   cd /workspace/CLIPGCN && \
   exec bash scripts/run_rosbot_stack.sh'
```

### 启动 VPOCLIP 图形窗口

`vpoclip_config.csv` 的 `display=false` 只用于后台/组合启动；桌面窗口入口会强制打开 GUI：

```bash
cd /home/youhan/HAR/HAR_rosbot_deployment
./scripts/start_vpoclip.sh
```

会出现两个窗口：`CLIPGCN rosbot_1` 和 `CLIPGCN rosbot_2`。窗口中按 `q` 或 `Esc` 可退出识别进程；四视角 DDQN 选择节点本身不创建窗口。

容器使用本机已有的 `clipgcn-rosbot:jazzy` 环境、NVIDIA GPU、ROS2 Jazzy 和 host ROS 网络。启动后包含以下节点：

- 每个 `HAR_VPOCLIP_ROBOT_NAMESPACES` 中的机器人各启动一个 `ros_realtime.py`，加载 `har_export` 的 VPOCLIP、X3D、CTR-GCN、YOLO 和 RTMPose 做实时识别。
- `rosbot_1` 窗口保留本机当前视角的识别结果，并用洋红色叠加显示 `/har/final_action` 的双视角融合结果；融合结果来自两个机器人最新的 VPOCLIP 预测。
- `active_view_ros.py`：加载 `full_depth19` 双机器人 Pair-Dueling DDQN 和 RTMPose 方向 MLP，订阅四个视角并发布 `/har/selected_view_pair`。

## 三个启动脚本

从主机桌面终端执行下面三个入口。每个入口都会使用 Docker；Nav2 只在已有的 `jazzy-rosbot` 容器中启动。

### 1. VPOCLIP 识别窗口

```bash
cd /home/youhan/HAR/HAR_rosbot_deployment
./scripts/start_vpoclip.sh
```

默认打开 `CLIPGCN rosbot_1` 和 `CLIPGCN rosbot_2` 两个窗口。主机必须允许 X11：脚本会执行 `xhost +si:localuser:root`。
当前配置让两套 VPOCLIP 的 RTMPose 也使用 CUDA；双机启动器会串行预热两个识别器，降低 6 GB 显卡冷启动时的冲突风险。

### 2. Nav2 大模块

```bash
./scripts/start_nav.sh
```

它会在 `jazzy-rosbot` 中用 `home` 地图启动两台机器人的定位和 Nav2；需要 RViz 时使用 `./scripts/start_nav.sh --rviz`。重新启动导航使用 `./scripts/start_nav.sh --restart`。地图也可以在 `config/orbit_system_config.csv` 的 `map_name` 行调整。

默认情况下不需要分别填写两个机器人的初始坐标：`initial_robot_pose_mode=relative_human_slots` 会用人体坐标、人体朝向、`ring_radius_m` 和 `initial_robot_slots=0;1` 自动计算 rosbot_1/rosbot_2 的启动 `x/y/yaw`，且两个机器人都会朝向人体。这里的机器人坐标和 slot 绑定只是第一次启动/第一次规划前的 bootstrap，不是永久约束；进入真实运行后，每次规划优先读取 `map -> rosbot_i/base_link` 的当前 TF，并在机器人到达目标后更新当前 slot。真实导航进程已经运行时，修改人体姿态后需要用 `--restart` 重新加载定位初始位姿；普通启动不会强行改变已经运行的定位：

```bash
./scripts/start_nav.sh --dry-run
./scripts/start_nav.sh --restart --rviz
```

如果需要兼容旧的人工测量方式，把 `initial_robot_pose_mode` 改成 `manual_csv`，再编辑 [`config/robot_initial_poses.csv`](./config/robot_initial_poses.csv) 中对应地图、机器人和 `enabled=true` 的行。`yaw_deg` 用角度填写，脚本会自动转换成 ROS 所需的弧度。

### 3. 三选一的视角移动策略

一次只运行一个模式：

```bash
./scripts/start_motion.sh rl       # har_export 的 full_depth19 DDQN
./scripts/start_motion.sh random   # random_pair
./scripts/start_motion.sh cyclic   # cyclic_pair：0-2、1-3 循环
```

这三个模式严格对应 `har_export` 的强化学习、随机和 cyclic 视角选择协议，结果发布到 `/har/selected_view_pair`。当前导出包只有“选择视角”的 RL 权重，没有地图坐标到 Nav2 goal 的底盘运动权重，因此该脚本不会直接发布 `cmd_vel`；真实底盘导航由第二个脚本和 Nav2 目标负责。

## 机器人移动/视角选择配置表

文件：[`config/orbit_system_config.csv`](./config/orbit_system_config.csv)

这张表是机器人移动系统的统一入口，格式和 VPOCLIP 的参数表一样，修改后重启相关脚本即可生效。常用字段如下：

```text
selection_mode=rl/random/cyclic       选择导出的 DDQN、随机或 cyclic 视角策略
dry_run=true/false                    true 只打印目标坐标；false 才允许发送 Nav2 goal
ring_radius_m=1.5                    以当前人体点为中心的圆半径，单位 m
ring_angles_deg=270;0;90;180         四个相对人体正面的 slot 偏移；slot 1 为正前方 0°
human_position_mode=fixed             手动人体地图坐标；不使用深度图或人体像素反投影，可由 RViz 在线更新
human_x_m=0.0 / human_y_m=0.0          当前人体点在 map 中的 x/y（这是启动值），按现场位置修改
human_z_m=0.0                          固定人体点 z；仅用于诊断，不影响二维导航
human_yaw_deg=0.0                      人体在 map 中的朝向；ROS 角度约定，+x 为 0°
human_front_slot=1                     人体正面对应 slot 1
initial_robot_pose_mode=relative_human_slots
初始机器人位姿由人体和 slot 自动推导；可改为 manual_csv。仅作为启动基准，不锁定后续机器人位置
human_marker_topic=/har/orbit/human_center_marker
                                      RViz 中显示当前人体点的红色 Marker
human_heading_marker_topic=/har/orbit/human_heading_marker
                                      RViz 中显示人体朝向的黄色箭头
human_pose_topic=/har/orbit/human_pose
                                      RViz 在线人体姿态工具发布的话题
initial_robot_marker_topic=/har/orbit/initial_robot_markers
                                      RViz 中随人体移动的两个虚拟初始机器人箭头/标签
initial_robot_slots=0;1               仅表示初始 slot；真实移动后由到达目标更新，不是永久绑定
bootstrap_pair=0;2                    没有收到策略结果时使用的初始目标槽位
active_view_variant=full_depth19     导出的 DDQN 结构
active_view_policy_device=cuda       DDQN 运行设备
active_view_rtmpose_device=cuda      视角策略单独使用的 RTMPose 设备
fusion_continuous=true               是否持续融合两个最新 VPOCLIP 结果
```

完整系统入口如下。它会启动已有 Nav2、RViz、Follow、双机 VPOCLIP、融合、视角策略和环形协调器；Follow 在机器人移动时关闭、静止时开启。配置表默认 `dry_run=true`，所以第一次只输出 rosbot_1 先走、rosbot_2 后走的目标和轨迹，不会移动底盘：

dry-run 不会把模拟到达的目标写回真实机器人槽位；`dry_run_update_slot_mapping=false` 时，主动视角模块仍按机器人实际所在槽位工作。若任一机器人相机超过 `active_view_input_stale_seconds`（默认 3 秒）没有新帧，视角选择会暂停，避免断线时复用旧的 13 帧历史。

```bash
./scripts/start_orbit_system.sh
```

完整系统停止使用一键脚本。它会停止 HAR/VPOCLIP、Follow、融合、视角选择、定位、Nav2 和本项目启动的 RViz，但不会删除 Docker 容器、镜像或数据卷：

```bash
./scripts/stop_har_system.sh
```

命令行参数可以临时覆盖表中的选择模式，但不会修改 CSV：

```bash
./scripts/start_orbit_system.sh --selection random
./scripts/start_orbit_system.sh --selection cyclic --continuous
```

`start_motion.sh` 不带参数时也会读取 `selection_mode`；带 `rl`、`random` 或 `cyclic` 时只对本次启动覆盖它。实际的 map 坐标规划和 Nav2 串行发送由 `orbit_coordinator.py` 完成，启动时使用 `human_x_m/human_y_m`，运行中可由 RViz 的 `/har/orbit/human_pose` 更新环心和朝向；系统不订阅深度图，视角策略本身只负责选择下一个环形槽位。

### 在 RViz 在线调整人体和两个机器人显示位置

RViz 的 `Fixed Frame` 保持为 `map`。工具栏现在有两个 `2D Pose Estimate`：原来的工具仍用于设置 rosbot_1 定位；另一个发布到 `/har/orbit/human_pose`，用于人体。对人体工具在地图上单击人体中心并拖动箭头设置朝向；RViz 内置工具会在松开鼠标时发布姿态。协调器会立即更新红色人体球、黄色人体朝向箭头，以及蓝色/绿色的两个“virtual initial robot poses”箭头和标签；这两个虚拟箭头会跟随人体坐标和朝向移动。

这里的虚拟机器人显示不会覆盖真实 `map -> rosbot_i/base_link` TF，也不会因为拖动人体而让底盘自行移动。RViz 中真实机器人模型仍表示机器人当前真实定位；蓝色/绿色箭头表示“按 slot 0/1 推导出来的初始相对坐标”，可以用来检查配置是否正确。`persist_rviz_human_pose=true` 时，拖动结束的 x/y/z/yaw 还会写回 `config/orbit_system_config.csv`，作为下次启动值。若要让 Nav2 重新采用这两个推导出的初始位姿，确认现场安全后执行 `./scripts/start_nav.sh --restart`。

四个环形目标由 `/har/orbit/target_poses` 显示。人体朝向改变时，slot 0/1/2/3 的 map 坐标会随之旋转；默认 `270;0;90;180` 表示 slot 1 在人体正前方，slot 0/2 在左右两侧，slot 3 在后方。若 RViz 窗口已经打开，重启 RViz 后才能读取新增的工具和显示项。

确认 RViz 中的目标和日志都正确后，才把表中的 `dry_run` 改为 `false`，或者明确执行 `./scripts/start_orbit_system.sh --live`。移动期间 Follow 保持关闭，rosbot_1 完成后才允许 rosbot_2 开始；机器人不在移动时 Follow 自动开启，但它只发布原地转向：Follow 控制器固定 `linear.x=0`，只使用 `angular.z` 调整视角，不会让机器人平移。最终 `cmd_vel` 仍由 arbiter 仲裁，Nav2 移动优先级高于 Follow。双机 VPOCLIP 融合本身始终使用两边最新结果，不等待移动完成。`settle_seconds=3.0` 仍用于落位后的识别周期/主动视角反馈标记。

## 下次启动

推荐使用一个总入口。第一次验证目标坐标时使用持续 dry-run：

```bash
cd /home/youhan/HAR/HAR_rosbot_deployment
./scripts/start_orbit_system.sh --dry-run --continuous
```

确认 RViz、目标坐标和日志都正确后，真实移动时明确使用：

```bash
./scripts/stop_har_system.sh
./scripts/start_orbit_system.sh --live
```

`--live` 会发送 Nav2 移动目标并驱动机器人，执行前请确认场地安全。地图默认从 `config/orbit_system_config.csv` 的 `map_name=home` 读取。

实时查看人体中心、四个环形候选点、两台机器人的目标和中间 waypoint，另开一个终端执行：

```bash
./scripts/watch_orbit_targets.sh
```

这是只读监视器，按 `Ctrl-C` 只退出监视器，不会停止机器人系统。它显示的坐标均为 `map` 坐标系；dry-run 时会明确显示 `NO NAV2 GOAL SENT`。

默认 `HAR_VPOCLIP_ROBOT_NAMESPACES=rosbot_1,rosbot_2`，因此两台机器人会同时启动独立的 VPOCLIP ROS 节点；显存不足时可改成单机器人：

```bash
export HAR_VPOCLIP_ROBOT_NAMESPACES=rosbot_1
docker compose up -d
```

双机 CUDA 模型会串行初始化，避免两套 RTMPose/YOLO 在冷启动瞬间争抢显存；等待上限由 `HAR_VPOCLIP_START_TIMEOUT_SEC` 控制，默认 120 秒。

当前全 GPU 配置会让两个 VPOCLIP 识别器、主动视角策略和各自的 RTMPose 都使用 CUDA；如果显存不足，可临时把 `rtmpose_device` 或 `active_view_rtmpose_device` 改回 `cpu`。

## 两张可编辑配置表

### 1. 动作候选表

文件：[`realtime_class_config.csv`](./realtime_class_config.csv)

表格格式与 `VPOCLIP_rosbot/realtime_class_config.csv` 一致：

```text
enabled=true/false   是否参与 VPOCLIP ranking；false 会从候选池完全移除
split=seen/unseen    归入 seen 或 unseen 候选侧
score_scale          该类分数乘法校准，通常保持 1
score_bias           该类分数加法校准，通常保持 0
```

当前默认表对应 `RESULTS.md` 的 `strict45_10`：55 个模型类别全部启用，10 个 unseen 是 `1,7,14,15,18,25,39,46,52,54`；其中最终测试 unseen 是 `25,39,46,52,54`。动作 ID 是从 0 开始的模型 label，`A01` 对应 label `0`。

例如禁用“reading a book”，只改这一行的 `enabled`：

```text
A32,31,reading a book,false,seen,1,0
```

改完后重新启动 VPOCLIP。`candidate_scope=all` 时会在表中所有 enabled 的 seen/unseen 类别上 ranking；如果把某类改成 `enabled=false`，它不会再参与 ranking。

### 2. VPOCLIP 参数表

文件：[`vpoclip_config.csv`](./vpoclip_config.csv)

这张表把 `ros_realtime.py` 的模型、权重、RTMPose、时间窗口、ranking、熵门控、YOLO、显示和跟随输出参数集中在一起。启动脚本会自动读取它，路径使用当前项目根目录作为相对路径基准。常用修改例如：

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

`start_vpoclip.sh` 专门用于打开窗口，会强制 `display=true`；它仍会读取其余参数。当前双机器人默认使用 YAML 中的 CUDA 配置运行 VPOCLIP/YOLO/特征模型，RTMPose 也使用 CUDA。两个识别器会串行初始化，并在启动第二个前等待第一个稳定；若只想临时覆盖，可设置 `HAR_VPOCLIP_RTMPOSE_DEVICE=cuda`。若把 `runtime_device` 改为 `cuda`，程序会按全 CUDA 模式运行并覆盖独立的 RTMPose 设备设置。

其中 `runtime_device=config` 是本项目的特殊值：表示不额外传入 `--runtime-device`，直接使用 `config_runtime.yaml` 的 `runtime.device`，从而允许 `rtmpose_device` 单独设为 CPU。

表中路径来自当前新项目的 `vendor/`，不是旧容器中的 `/workspace/CLIPGCN/work_dir/...`；修改 checkpoint 时必须同时确认 `config`、`class_split_dir` 和 CTR-GCN/X3D 的结构仍然匹配。

默认相机主题是：

```text
/rosbot_1/camera/rgb/image_raw/compressed
/rosbot_2/camera/rgb/image_raw/compressed
/har/view/slot_2/compressed
/har/view/slot_3/compressed
```

如果主题不同，启动前设置：

```bash
export HAR_VIEW_TOPICS=/你的主题0,/你的主题1,/你的主题2,/你的主题3
export HAR_ROBOT_NAMESPACE=rosbot_1
docker compose up -d
docker exec -d har-rosbot-deployment bash -lc \
  'source /workspace/CLIPGCN/scripts/runtime_env.sh && \
   cd /workspace/CLIPGCN && exec bash scripts/run_rosbot_stack.sh'
```

查看输出：

```bash
docker exec har-rosbot-deployment bash -lc \
  'source /opt/ros/jazzy/setup.bash && ros2 node list && ros2 topic list'
docker exec har-rosbot-deployment bash -lc \
  'source /opt/ros/jazzy/setup.bash && ros2 topic echo /har/selected_view_pair'
tail -f logs/active_view.log logs/recognizer_rosbot_1.log logs/recognizer_rosbot_2.log
```

## 使用已有的 jazzy 导航容器

`jazzy-rosbot` 是独立的导航容器；不要把 Nav2 启动到 HAR 容器里。两台机器人使用已有 `home` 地图定位并启动 Nav2：

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

然后在 HAR 容器中启动两台机器人的速度仲裁器：

```bash
for ns in rosbot_1 rosbot_2; do
  docker exec -d har-rosbot-deployment bash -lc \
    "source /workspace/CLIPGCN/scripts/runtime_env.sh && \
     exec ros2 launch person_follow_demo person_follow_demo.launch.py \
     namespace:=$ns >>/workspace/CLIPGCN/logs/follow_${ns}.log 2>&1"
done
```

最后执行上面的 HAR Docker 启动命令。Nav2 发布到 `/<robot>/cmd_vel_nav`，仲裁器是最终 `/<robot>/cmd_vel` 的唯一发布者；验证阶段不要执行 `rosbot-offboard goal`，确认 RViz 中定位正确后再发送目标。

## 权重对应关系

权重都保存在 `vendor/`，没有重新下载：

| 模块 | 文件 |
|---|---|
| VPOCLIP | `vendor/HAR_reproduction/ws/VPOCLIP_plus_full/work_dir/merged45_10_groupAB/selected_stage_b/last_model.pth` |
| X3D | `vendor/HAR_reproduction/ws/X3D_full/outputs/etri_allviews_cs45_merged_groupAB_7000/model_best.pth` |
| CTR-GCN COCO-17 | `vendor/HAR_reproduction/ws/CTR-GCN_17_full/work_dir/etri_coco17/allviews_cs45_merged_groupAB/runs-51-4029.pt` |
| YOLO custom-50 | `vendor/HAR_reproduction/ws/yolov5_full/best.pt` |
| RTMPose | `vendor/HAR_reproduction/.cache/rtmlib/hub/checkpoints/` |
| 视角 DDQN | `vendor/HAR_reproduction/ws/VPOCLIP_plus_full/work_dir/dual_robot_depth_ddqn_strict45_5/models/full_depth19/seed_20260911/best.pt` |
| 方向 MLP | `vendor/RTMPose_orientation_MLP/work_dir/rtmpose_orientation_mlp/best.pt` |

## 跟随机器人

默认只做识别和发布结果。若已启动本机 `person_follow_ws` 中的跟随控制节点，可以显式打开检测输出：

```bash
export HAR_ENABLE_FOLLOW=1
docker compose up -d
```

这会发布 `/<robot>/follow/person_detection`，不会自动启动底盘控制器。

## 重要的安全边界

导出的 DDQN 使用四个视角槽和归一化角度代价；导出包没有当前地图中的 TF、导航目标点或视角槽到 Nav2 goal 的标定。因此新节点只发布视角选择结果，不直接发送 Nav2 运动命令。确认现场 TF、地图和代价定义后，才应在 `/har/selected_view_pair` 后面接入站点专用的导航适配器。

没有 ROS 相机时，可先运行 `run_policy_smoke.sh` 验证 DDQN、方向 MLP、GPU 和权重加载；ROS 节点会等待真实图像主题。
