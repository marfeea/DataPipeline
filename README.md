# R2Dreamer IK 数据产线

当前真实采集入口使用 `r2dreamer-aubo-ik-aligned-v4`。新旧数据使用独立输出目录。
采集命令指定已有 v3 目录时，会自动选择同级后缀 `_v4` 的目录，并打印 `DATASET_OUTPUT_REDIRECT`；
例如 `data/pick_place_smoke_4_to_0` 会写到 `data/pick_place_smoke_4_to_0_v4`。
重跑同一命令会续采该 v4 目录。历史目录不迁移、不覆盖；目标目录冲突或 manifest 损坏时仍报错。
本轮双坐标、夹爪力、commanded/applied、reward mask 和验收记录见
[数据规范对齐施工进度](docs/数据规范对齐施工进度_20260920.md)。
下文的 v3/P0 说明保留为历史接口背景。

v4 验收与完整回放可重复执行（在仓库根目录，使用 Isaac Python）：

~~~powershell
python scripts/audit_alignment_dataset.py ../data/alignment_v4_acceptance_batch_20260920 --legacy-dataset data/pick_place_smoke_0_to_1 --report-json ../data/alignment_v4_acceptance_audit.json
python scripts/replay_aubo_ik_trajectory.py --output-dir ../data/alignment_v4_pick_place_20260920_r2 --all-episodes --mode action --headless --frame-delay 0 --report-json ../data/alignment_v4_full_action_report.json
~~~

验收报告包含独立 validator 结果、必需任务元数据、T/T+1 字段形状、commanded/applied 差异和训练 reward mask 检查。
回放报告包含完成帧数与关节、夹爪、TCP 位姿的最大/RMS/末帧误差；完成状态表示请求帧已执行，
不等于物体成功判据。项目尚未规定回放精度阈值，因此报告保留测量值，不自动宣称精度达标。
v4 训练须将 batch 的布尔 reward_valid 传入 reward loss；缺失、非布尔或错误形状的 mask 会被 reader 拒绝。


本仓库是从 `CurriculumRL/datapipe` 迁出的独立数据项目。实现位于
`DataPipe/`，Python 导入名统一为 `DataPipe`。原 Isaac Lab 模板的
`DataPipe/tasks/` 与 UI 示例保留，但不再由顶层包自动注册。

DataPipe 拥有 episode/schema、HDF5 writer/reader/validator、序列规划、采集适配、
重播与数据加载逻辑。全部配置常量已复制到 `configs`，包括目标状态、相机标定、机器人和控制参数。
AUBO FK/IK、正式环境、奖励和终止实现仍由外部 `CurriculumRL` 提供。完整边界见
[项目边界](docs/项目边界.md)。

## 目录结构

```text
DataPipe/       核心 Python 包（导入名 DataPipe）
scripts/        采集、回放、数据校验与运行脚本
configs/        场景、相机、控制等数据采集配置
tests/         单元测试和测试用脚本（Isaac smoke 脚本在 isaac/）
docs/           项目文档
```

从本仓库根目录运行 `python scripts/collect_aubo_ik_dataset_isaac.py ...`。
普通测试运行 `python -m pytest tests`；`tests/isaac/` 内的脚本需在 Isaac Lab 环境手动运行。
原 RL 训练/回放模板仍保留在 `scripts/` 的框架子目录中。

## 安装

在本仓库根目录执行：

```powershell
$isaac = "D:\Anaconda\envs\isaaclab\python.exe"
& $isaac -m pip install -e ..\..\CurriculumRL\source\CurriculumRL
& $isaac -m pip install -e .
```

采集与回放入口会自动发现相邻的 `S2R/CurriculumRL` 源码，无需先安装该包。
其他目录布局可设置 `$env:DATAPIPE_CURRICULUMRL_ROOT = "<CurriculumRL仓库路径>"`。
配置快照来自相邻仓库（2026-09-05），不会自动同步；外部环境内部仍使用自身配置，
调整本地参数时应确认与外部环境兼容。USD 资产仍从外部 Asset 目录读取。

当前已完成 P0 契约和数据闭环骨架：

1. 固定 episode 时序：观测长度为 `T+1`，动作和奖励长度为 `T`，严格组织为
   `o0-a0-r0-o1-...-oT`。
2. 固定边界语义：首观测写 `is_first`；成功和安全失败写 `is_last + is_terminal`；超时只写 `is_last`；终局观测必须在 reset 前显式提交。
3. schema `r2dreamer-aubo-ik-sequence-v3` 固定 8 维动作 `[dx,dy,dz,rx,ry,rz,tcp_heading_target,gripper]`；六轴 IK 与两轴夹爪仍由独立执行适配器消费，`tcp_heading_target` 记录 XOY 平面 TCP 朝向终端值。
4. P0-4～P0-7：登记现有静态 IK 场景基线、相机位姿门禁、正式环境终局结果适配、唯一 episode ID 与 state/action replay 选择。
5. P2 纯逻辑骨架：轨迹规划、采集快照、原子 JSONL 分片、manifest 校验和、断点续采、validator 和 batch 适配；纯逻辑部分仅用于单元测试，不作为最终采集入口。

## 多阶段采集状态

采集器只支持 `pick-place`，预停靠后执行 `ALIGN_NORMAL -> ADVANCE_NORMAL -> CLOSE_GRIPPER`，随后继续 `LIFT_OBJECT -> PLACE_HIGH_XY -> ALIGN_PLACE_DIRECTION -> PLACE_FINAL_Z -> PLACE_FINAL_XY -> RELEASE_GRIPPER -> PLACE_COMPLETE`。全程由同一 Flange-down 约束保持 Flange 法向朝下；`ALIGN_NORMAL` 在预停靠位置按最短角度分支限步长对齐抓取目标的 TCP XOY 朝向，位置与朝向同时满足 dwell 后才前进，`ADVANCE_NORMAL` 复用固定 TCP 姿态完成位移，TCP 的 XOY 平面朝向另作为第 7 个动作分量记录。高位搬运沿最短角度分支同步过渡到源/目标 heading 的中点，到达目标上方后再完成剩余转向；这避开“源上方转完整角度”的 q6 限位分支和“全程保持源 heading”的局部 IK 死点。阶段只由实测位置、TCP 平面朝向、夹爪位置和 dwell 门控。正式 RL 任务配置不变，只有采集环境副本隔离旧停车成功与“目标移动”终止。

GUI replay 启动后会读取 `configs.render.TEST_RENDER_CFG`，将 viewport camera 设置为项目视角：eye `(-0.5, 0.2, 1.25)`，target `(0.5, 0.2, 1.25)`；这与 CameraSensor 三路观测相机是两套独立配置。

## CameraSensor 位置

数据管线通过 `project_diagnostic_camera_sensor_specs()` 读取
`configs.vision_diagnostics.DIAGNOSTIC_CAMERA_POSES`，不再使用
`TEST_RENDER_CFG` 的 viewport 相机。三路环境坐标位置固定为：

- `camera_cfg` 前视：`(2.7, 0.0, 1.3) m`；
- `camera_cfg_2` 侧视：`(1.3, -0.8, 2.0) m`；
- `camera_cfg_3` 俯视：`(1.3, 0.0, 2.8) m`。

真实 Isaac 采集入口会同时挂载三路 CameraSensor，并在每个控制步检查三路帧集合、
仿真步和时间戳；观测按 `camera/<camera_name>/{rgb,distance_to_image_plane}`
组织。CLI 默认采集分辨率仍为每路 `160x120`，可通过参数覆盖；焦距、孔径、裁剪范围
和三路外参继续以 `configs/vision_diagnostics.py` 为唯一事实源。

## 使用

```powershell
python -m pytest -q tests
$isaac = "D:\Anaconda\envs\isaaclab\python.exe"
& $isaac scripts/collect_aubo_ik_dataset_isaac.py --output-dir <OUTPUT_DIR> --episodes 1 --target-state-index 0 --sequence-mode pick-place --place-state-index 1 --lift-clearance-m 0.40 --gripper-closed-position 0.0 0.0 --log-sequence-stage-transitions --headless
& $isaac scripts/validate_aubo_ik_dataset.py <OUTPUT_DIR>
$episode = ((Get-Content "<OUTPUT_DIR>\manifest.json" -Raw | ConvertFrom-Json).episode_ids[0])
& $isaac scripts/replay_aubo_ik_trajectory.py --output-dir <OUTPUT_DIR> --episode-id $episode --mode state
& $isaac scripts/replay_aubo_ik_trajectory.py --output-dir <OUTPUT_DIR> --episode-id $episode --mode action

# 从源 state 抓取后，抬升、高位 XOY 平移到目标 state、Z 轴对齐并松爪
& $isaac scripts/collect_aubo_ik_dataset_isaac.py --output-dir <NEW_OUTPUT_DIR> --target-state-index <SOURCE_INDEX> --sequence-mode pick-place --place-state-index <DESTINATION_INDEX> --lift-clearance-m <CLEARANCE_M> --gripper-closed-position <UPPER> <DOWN> --headless
```

真实 Isaac 采集入口参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--output-dir` | 必填 | 数据集输出目录。 |
| `--episodes` | `3` | 本次目标 episode 数量。 |
| `--max-steps` | `960` | 单 episode 的采集上限；抓取序列的采集专用环境时限随该值设置，12 Hz 下为 80 秒；不修改正式 RL 环境配置。 |
| `--waypoint-position-tolerance-m` | `0.01` | 实际 TCP 到达当前 waypoint 的推进容差，单位 m。 |
| `--seed` | `7` | 仿真与确定性轨迹种子。 |
| `--target-state-index` | `0` | `TARGET_STATES` 中的目标索引，范围 `0..4`。 |
| `--run-tag` | `isaac-ik-batch` | episode ID 的运行标签前缀。 |
| `--planning-hz` | `4.0` | 粗规划频率。 |
| `--control-hz` | `12.0` | 轨迹控制频率，必须映射为整数物理步。 |
| `--coarse-min-waypoints` / `--coarse-max-waypoints` | `30` / `40` | 粗规划点数量范围。 |
| `--planning-waypoints` | `None` | 固定粗规划点数量；未指定时由 seed 采样。 |
| `--xy-fraction-min` / `--xy-fraction-max` | `0.40` / `0.60` | staged 轨迹的平面移动比例范围。 |
| `--trajectory-mode` | `staged` | `direct` 或 `staged`，动作始终来自共享 IK 轨迹。 |
| `--high-clearance-z` | `None` | staged 轨迹的 B 系高位 Z。 |
| `--base-clockwise-deg` | `45.0` | staged 基座预旋转角。 |
| `--downward-yaw-offset-deg` | `0.0` | Flange 向下姿态的附加 yaw，单位度。 |
| `--sequence-mode` | `pick-place` | 唯一支持的采集模式。 |
| `--normal-advance-distance-m` | `0.075` | `task.py` 中配置的 preposition 到闭爪抓取 TCP 点的距离，单位 m；显式传入时必须保持为该值。 |
| `--place-state-index` | `None` | `pick-place` 必填的目标 state 索引；放置 TCP 位置取该 state 的 `grasp_tcp_position_e`，索引范围 `0..4`。 |
| `--lift-clearance-m` | `None` | `pick-place` 必填的正抬升间隙，单位 m。 |
| `--normal-angle-tolerance-deg` | `5.0` | 抓取与放置的 TCP 平面 heading 到位容差，单位度。 |
| `--normal-dwell-steps` / `--gripper-dwell-steps` | `3` / `3` | 姿态/位置与夹爪到位所需连续步数。 |
| `--gripper-position-tolerance` | `0.008` | 两夹爪关节到位容差。 |
| `--gripper-open-position` | `0.0115 -0.0115` | `UpperFinger DownFinger` 张开目标。 |
| `--gripper-closed-position` | `None` | 必填的两个闭合目标；启动时按 USD 限位校验。 |
| `--camera-width` | `160` | 每路相机图像宽度，像素。 |
| `--camera-height` | `120` | 每路相机图像高度，像素。 |
| `--kit-cache-dir` | `data/isaac_kit_cache` | Kit 用户配置与 OptiX 缓存目录。 |

入口强制启用 CameraSensor；其他 Isaac Lab 公共启动参数由 `AppLauncher` 提供。

所有阶段使用 `flange_down_6d`；进入 `ADVANCE_NORMAL` 前完成 Z/位置及抓取 TCP XOY 朝向对齐，随后沿抓取 TCP 终点位移并保持 TCP 朝向。第 7 维 `tcp_heading_target` 是 XOY 平面朝向的归一化终端值；`PLACE_HIGH_XY` 在搬运时过渡到源/目标 heading 的角度中点，`ALIGN_PLACE_DIRECTION` 到达目标上方后完成剩余转向，两阶段沿角度 wrap 后的最短分支每步最多改变 5°，并通过考虑 TCP 偏置的迭代几何求解转换为 Flange-down yaw。前六维仍提交给原 action adapter，但该 adapter 在 Flange-down 模式下会把旋转增量置零，姿态由 heading 目标生成；IK 数学求解器本身未修改。夹爪命令由独立 adapter 映射到两个关节目标，不能进入六关节 IK。HDF5 记录 8 维归一化动作、`semantic_stage_id`、独立 `waypoint_index`、两夹爪关节状态，以及 `tcp_heading_target/actual/first_five/sixth_axis/error` 诊断。新旧 schema 必须使用不同输出目录，writer 会拒绝混写；validator 只接受当前 pick-place 阶段顺序。

action replay 从 episode 元数据恢复 `ik_constraint_mode` 和 yaw；没有这些字段的旧数据继续使用原 runtime curriculum IK，不会被静默切换到 Flange-down。

纯逻辑模块不依赖 Isaac Lab 或 Torch；真实采集入口需要 Isaac Lab 环境。三路相机配置
和采集适配已接入代码，但仍需实际 Isaac 动态 smoke 才能确认渲染输出。GUI 可视化重播脚本
已接入同一动态场景，支持 state/action 两种模式；R2Dreamer/TensorDict 训练更新仍未接入。
正式采集记录环境总 reward 到 `rewards[T]`，并记录 `diagnostics/reward_components/*[T]`；
v4 将这些值保留为诊断，并设置 `reward_status=not_defined`、`reward_valid=false`。
训练使用 `DataPipe.reward.compute_reward_prediction_loss` 时必须传入 batch 的 `reward_valid`；
占位奖励不会贡献损失。旧 v3 数据保持原有奖励监督语义。

字段所有权与外部配置依赖见 [项目边界](docs/项目边界.md)。

## Reward 预测损失

`DataPipe.reward.compute_reward_prediction_loss()` 提供参考
`mamba_world_model.py` reward head 的纯 NumPy 监督损失：默认使用逐元素
Smooth L1；传入 `RewardLossConfig` 时，可按高奖励阈值和奖励值对样本加权。
该函数计算的是世界模型的 reward prediction loss，不替代正式任务逻辑产生的
奖励或终止判定。

```python
from DataPipe import RewardLossConfig, compute_reward_prediction_loss

loss = compute_reward_prediction_loss(
    predicted=[0.2, 0.8],
    target=[0.0, 1.0],
    config=RewardLossConfig(positive_weight=2.0),
)
```
