# CuRobo NPZ 场景一致可视化重播 MVP 施工方案

日期：2026-09-22。

> 本文是 CuRobo 可视化重播的唯一施工方案：规划场景一致性属于回放硬门；现有 run_006 产物直接作为本文第 7 节的验收标准数据基础，不另立验收标准文档。

## 1. 本轮唯一目标

提供一条命令，读取 CuRobo MVP 的 `trajectory.npz` 及其同次规划 `report.json`，在 Isaac Lab 中重建同一机器人、同一 Base 坐标契约和同一静态世界，以 60 Hz 逐帧渲染全部轨迹状态，并输出场景一致性与回放完成报告。

本轮交付为：

- 唯一可运行回放命令；
- 与 CuRobo 规划场景一致的 Isaac Lab 最小场景；
- GUI 60 Hz 全帧 state replay；
- `scene_consistent=true` 且 `passed=true` 的 `replay_report.json`；
- 纯逻辑测试、headless 集成 smoke 和一次真实 GUI 验收。

## 2. MVP 技术决策与边界

### 2.1 选择 Isaac Lab 专用场景

首版使用 Isaac Lab，但不直接复用包含完整实验室资产的 `CurriculumRL-TcpDocking-Trajectory-v0`。新增专用最小 replay scene，只允许：

- 一台被规划的 AUBO；
- 本次规划报告中的全部静态 world primitive；
- 无碰撞的灯光、viewport 相机和调试标记。

禁止额外加入工作站、地面、托盘、样品瓶、第二台机器人或其他 collision-enabled prim。若需要这些实体，必须先将其纳入 CuRobo world、重新规划并形成新的 world hash。

### 2.2 首版采用 state replay

- 每帧直接写入 NPZ 中的 `joint_position` 和 `joint_velocity`；
- 同步 articulation position target，随后执行 `scene.write_data_to_sim()`、`sim.forward()`、场景更新和一次显式 `sim.render()`；
- 不推进物理跟踪，不把视觉回放冒充 action/physics replay；
- `joint_acceleration` 本轮只校验和报告，不写入 Isaac；
- 夹爪保持固定开位，不属于当前六轴 CuRobo 轨迹。

### 2.3 60 FPS 的定义

本轮 60 FPS 指：

1. NPZ 相邻 `time_s` 等于 `1/60 s`；
2. 每个源采样点恰好触发一次显式 render；
3. GUI 使用 `start_monotonic + time_s[i]` 的绝对 deadline；
4. 不插值、不重采样、不跳帧、不重复帧；
5. 最终 `source_frames = written_frames = render_calls`。

实际屏幕刷新受 GPU、窗口合成器和显示器影响，因此报告实际 wall-clock FPS、迟到帧和最大延迟；严格 60 FPS 视频编码留到后续离屏采帧阶段。

### 2.4 明确不做

- 不重新调用 CuRobo，不在线规划；
- 不接受非 60 Hz NPZ；
- 不 clipping 或静默重排关节；
- 不做动力学跟踪精度、连续扫掠碰撞或生产场景验收；
- 不接抓取、夹爪事件、attach/detach、HDF5 或视频导出；
- 不修改旧 `replay_aubo_ik_trajectory.py` 的 HDF5 state/action 语义。

## 3. 规划场景一致性硬门

只有以下五层全部通过，才允许播放第一帧。

### 3.1 规划产物身份一致

- `report.json.passed=true` 且 `stage=complete`；
- 报告中的 `trajectory_path` 与请求 NPZ 匹配；
- NPZ SHA256 与本文第 7 节 golden sample 一致；
- 报告中的 control rate 与 NPZ 时间步同时为 60 Hz；
- NPZ 使用 `allow_pickle=False` 重新读取并校验，不能只信任旧报告。

### 3.2 机器人模型一致

- 派生 URDF 与 collision model 实算 SHA256 必须匹配规划报告；
- NPZ、URDF、MotionGen 报告和 Isaac articulation 的六轴名称与顺序完全一致；
- Isaac 实测位置/速度限位与 MotionGen runtime limits 在 `1e-6` 容差内一致；
- q0 与 qN 的派生 URDF/Isaac FK 位置误差 `<= 1e-5 m`、旋转误差 `<= 1e-3 rad`。

Isaac 使用视觉 USD，CuRobo 使用派生 URDF 和球体碰撞模型；一致性由资产 hash、关节契约和 FK 实测建立，不宣称视觉 mesh 与碰撞球逐三角形相同。

### 3.3 Base/World 坐标一致

CuRobo world pose 全部表达于机器人 Base 系 `B`。Isaac 显式配置并实测 `T_WB`，规划对象使用：

```text
T_WO = T_WB @ T_BO
```

生成对象后再回读 Isaac 实际 World pose，并计算：

```text
T_BO_measured = inverse(T_WB_measured) @ T_WO_measured
```

禁止直接把 Base 系数值当 World 系数值，除非实测证明 `T_WB` 为单位变换。Base pose 平移误差门为 `1e-6 m`，旋转误差门为 `1e-6 rad`，四元数固定 wxyz 且必须为单位四元数。

### 3.4 静态世界一致

- 从规划报告恢复全部 cuboid 的 name、`pose_b` 和 `dimensions_m`；
- 使用 `WorldConfig`/`WorldSnapshot` 重算 world hash；
- Isaac 回读后的对象名称集合必须精确相同；
- 每个对象 Base 系位置误差 `<= 1e-6 m`；
- 旋转误差 `<= 1e-6 rad`；
- 尺寸误差 `<= 1e-6 m`；
- 播放期间 cuboid 与机器人 Base 不得发生超差漂移。

### 3.5 无额外规划障碍物

播放前列出 collision-enabled prim，只允许 AUBO 自身碰撞体和规划 world 中的 cuboid。出现任何未登记碰撞体时设置 `scene_consistent=false` 并拒绝播放。

## 4. 输入契约

正式输入不是孤立 NPZ，而是：

```text
trajectory.npz
+ 同次 plan report.json
+ report 指向的派生 URDF
+ report 指向的 collision model
+ 固定 target_name/world 语义
```

NPZ 字段必须精确为：

| 字段 | 形状 | 单位 | 用途 |
|---|---:|---|---|
| `time_s` | `[N]` | s | 绝对帧时间 |
| `joint_position` | `[N,6]` | rad | articulation 位置 |
| `joint_velocity` | `[N,6]` | rad/s | articulation 速度 |
| `joint_acceleration` | `[N,6]` | rad/s² | 校验与报告 |
| `joint_names` | `[6]` | Unicode | 保序关节映射 |

Kit 启动前完成字段、shape、finite、时间轴、起终点、文件 hash 和 world hash 校验。Isaac 创建后、首帧前完成实际关节、限位、FK、Base pose、world pose/size 和额外碰撞体检查。

## 5. 计划新增内容

### 5.1 `DataPipe/pipelines/curobo_bootstrap/replay.py`

纯逻辑模块，负责：

- `load_replay_spec(trajectory_path, plan_report_path)`；
- NPZ/报告/资产 hash 交叉校验；
- 从报告恢复并验证 `WorldSnapshot`；
- Base/World pose 正反变换；
- 绝对 frame deadline；
- replay 与 scene consistency 指标结构。

不得导入 Isaac Lab、Torch 或 CuRobo。

### 5.2 `DataPipe/curobo_replay_scene.py`

Isaac 专用模块，负责：

- 构建单 AUBO、报告驱动 cuboid 的最小场景；
- 配置无碰撞灯光与 viewport；
- 暴露实际 Base pose、cuboid pose/size 和 collision prim 清单；
- 执行 q/dq state write、Isaac FK 测量和每帧场景不变量检查。

### 5.3 `scripts/replay_curobo_trajectory_isaac.py`

唯一运行入口，建议最小 CLI：

```text
--trajectory PATH       必填，trajectory.npz
--plan-report PATH      默认 NPZ 同目录 report.json，但文件必须存在
--replay-report PATH    必填，场景一致性与回放报告
--no-realtime           仅供 headless smoke
--headless/--device     AppLauncher 参数
--kit-cache-dir PATH    独立 Kit/OptiX 缓存目录
```

MVP 不提供 `--scene`、`--fps`、`--speed`、`--loop` 或起止帧参数，避免绕过固定场景与全帧验收。

### 5.4 测试与记录

- 新增 `tests/curobo_bootstrap/test_replay.py`；
- 使用主脚本 `--headless --no-realtime` 做真实 Isaac smoke；
- 更新 `docs/curobo_acceptance.md` 和施工进度，记录命令、报告及失败尝试。

## 6. 敏捷施工阶段

### P0：冻结输入与场景身份

- 以第 7 节 run_006 组合为唯一首验数据；
- 重算四个文件 hash 与 world hash；
- 冻结 `target_name=target`；
- 确认旧完整实验室场景不进入本轮 replay。

停点：输入不能形成唯一身份链时，不进入 Isaac 接线。

### P1：纯逻辑 ReplaySpec

- 完成 NPZ、报告、资产 hash、world hash 的交叉校验；
- 完成 Base/World 变换与绝对 deadline；
- 单测错误字段、错误报告、交换报告、错误 hash、修改 world、错误 joint order 和非 60 Hz 时间轴。

停点：所有错误必须在 Kit 启动前 fail-fast。

### P2：专用一致场景

- 创建一台 AUBO 和报告中的全部 cuboid；
- 排除额外 collision-enabled prim；
- 实测 Base pose、对象 pose/size 和 q0/qN FK；
- 输出逐项 `scene_consistency`。

停点：只有 `scene_consistent=true` 才能播放第一帧。

### P3：60 Hz 全帧循环

- frame 0 只作为 `t=0` 初态写入并渲染；
- frame `1..N-1` 使用绝对 deadline；
- 每帧写 q/dq、同步目标、forward、update、render；
- 每帧检查 Base/world 不漂移；
- 记录写回误差、实际 FPS、迟到帧和中断状态。

停点：headless no-realtime 完成 35/35 帧，且场景一致性全通过。

### P4：GUI 正式验收与收尾

- 使用第 7 节命令完成实时 GUI 重播；
- 检查 A—D 四级标准全部通过；
- 报告、命令和实际结果写回 CuRobo 验收及施工进度；
- 一次正式通过即结束本轮，不追加视频、循环或 physics replay。

## 7. 验收标准

### 7.1 标准验收数据基础

现有 run_006 产物正式作为本施工方案的 golden sample：

| 角色 | 相对 DataPipe 仓库根路径 | SHA256 |
|---|---|---|
| 轨迹 | `data/curobo_mvp/run_006/trajectory.npz` | `899eb467978c4d5215b0c8d3d2b7874150c3e3bef3b96f3d42a02bbe98dcbce0` |
| 规划报告 | `data/curobo_mvp/run_006/report.json` | `260de79fdc27b06224292fd5ce3c941ce0dc1958263039b1633d741af6547fdd` |
| 派生 URDF | `configs/curobo/assets/aubo_e5_project/aubo_e5_project.urdf` | `93bf0b06c284ca42ed689d91977964fc26e146187a084dad06beb752f68a29d8` |
| 碰撞模型 | `configs/curobo/assets/aubo_e5_project/collision_model.json` | `609e9184c7e4f0050172511d66de2a7cdf6de45b96112d45cf4d94fbb475b41f` |

四项构成不可拆分的验收组合。修改任一文件后不得沿用本轮通过结论。

当前规划报告未内嵌 `trajectory_sha256`，因此本文显式冻结 NPZ hash。下一版规划产物应增加 `trajectory_sha256` 并由回放器强制校验。

### 7.2 NPZ 冻结值

| 项目 | 标准值 |
|---|---|
| 帧数 | `35` |
| q/dq/ddq shape | `[35,6]` |
| 起止时间 | `0.0` / `0.5666666666666667 s` |
| 时间步 | `1/60 s`，`atol=1e-9, rtol=0` |
| 关节顺序 | `Joint1, Joint2, Joint3, Joint4, Joint5, Flange` |
| finite | time/q/dq/ddq 全部为真 |
| 起点 | `[0, -0.5235987901687622, 1.2217304706573486, 0.7853981852531433, 1.5707963705062866, 0]` |
| 终点 | `[0.09999998658895493, -0.5235987901687622, 1.2217304706573486, 0.7853981852531433, 1.5707963705062866, 0]` |

### 7.3 World 冻结值

```text
report.passed = true
report.stage = complete
target_name = target
world_hash = 13d929d26a31b5e238f641bfc64fbde2dc331ea4d4352e3cfbeb7b443e2e8706
```

`target_name` 当前没有显式序列化到 run_006 报告，本文将其冻结为 `target`。回放器必须使用该值恢复 `WorldConfig`，不得根据列表顺序推断。

| 名称 | Base 系 `pose_b`（xyz+wxyz） | `dimensions_m` |
|---|---|---|
| `probe` | `[-3,-3,-3,1,0,0,0]` | `[0.02,0.02,0.02]` |
| `target` | `[3,3,3,1,0,0,0]` | `[0.02,0.02,0.02]` |

### 7.4 A 级：数据与规划产物身份

全部通过：

- 四个文件存在且 SHA256 等于第 7.1 节；
- 报告状态、trajectory path、URDF/collision hash 正确；
- NPZ 字段、dtype、shape、finite、joint order、起终点符合冻结值；
- 时间从零开始、严格递增、全部间隔为 `1/60 s`；
- 使用冻结 target name 恢复世界后，重算 world hash 一致。

A 级必须在启动 Kit 前完成。

### 7.5 B 级：Isaac 机器人一致性

全部通过：

- 仅一台目标 AUBO articulation；
- 四方关节名称/顺序一致；
- Isaac 限位与 MotionGen runtime limits 在 `1e-6` 内一致；
- 35 帧 q 全部在实际限位内；
- q0/qN 的 URDF/Isaac FK 位置误差 `<=1e-5 m`、旋转误差 `<=1e-3 rad`；
- 实测 Base pose 与配置一致。

### 7.6 C 级：规划世界与 Isaac 场景一致性

全部通过：

- 两个 cuboid 均由冻结 world 构造；
- Base/World 正反变换正确；
- 名称集合精确为 `{probe,target}`；
- pose 位置误差 `<=1e-6 m`、旋转误差 `<=1e-6 rad`；
- size 误差 `<=1e-6 m`；
- world hash 相同；
- 无额外 collision-enabled prim；
- 播放期间 Base 与 world 无超差漂移。

C 级通过后才能设置 `scene_consistent=true` 并进入帧循环。

### 7.7 D 级：60 Hz 全帧重播

全部通过：

- frame 0 不重复执行；
- 35 个源帧、35 次写入、35 次显式 render；
- 不插值、不重采样、不裁剪、不跳帧；
- q 写回最大误差 `<=1e-6 rad`；
- GUI 正式验收 `realtime_enabled=true`；
- 记录 actual FPS、late frames 和最大 deadline lag；
- 播放无异常或中断，资源完整关闭。

### 7.8 最终判定

`passed=true` 等价于 A、B、C、D 四级全部通过。仅窗口中机械臂移动、脚本退出码为 0、完成循环或 headless smoke 成功，均不能单独作为正式通过证据。

run_006 只能验收其固定合成场景，不得用于证明生产实验室场景一致或安全。

## 8. `replay_report.json` 最低字段

报告至少包含：

```text
passed / stage / failure_reason
acceptance_fixture = curobo_mvp_run_006
trajectory_path / plan_report_path
trajectory_sha256 / plan_report_sha256
urdf_sha256 / collision_model_sha256
source_frames / written_frames / render_calls
source_duration_s / target_frame_rate_hz
realtime_enabled / achieved_fps
late_frames / max_deadline_lag_ms
scene_consistent
configured_base_pose_w / measured_base_pose_w / base_pose_error
planned_world_hash / reconstructed_world_hash
expected_cuboids / spawned_cuboids
cuboid_pose_max_error_m / cuboid_rotation_max_error_rad
cuboid_size_max_error_m
unexpected_collision_prims
fk_start_error / fk_goal_error
max_q_writeback_error_rad / max_dq_writeback_error_rad_s
root_pose_drift / world_pose_drift
mode = state
scope = state_replay_same_planning_scene
```

## 9. 正式验收命令

从 `D:/Project/S2R/DataPipeline/DataPipe` 运行 headless smoke：

```powershell
$isaac = "D:/Anaconda/envs/isaaclab/python.exe"
& $isaac scripts/replay_curobo_trajectory_isaac.py `
  --trajectory data/curobo_mvp/run_006/trajectory.npz `
  --plan-report data/curobo_mvp/run_006/report.json `
  --replay-report data/curobo_mvp/run_006/replay_headless_report.json `
  --device cuda:0 --headless --no-realtime
```

再运行 GUI 正式验收：

```powershell
& $isaac scripts/replay_curobo_trajectory_isaac.py `
  --trajectory data/curobo_mvp/run_006/trajectory.npz `
  --plan-report data/curobo_mvp/run_006/report.json `
  --replay-report data/curobo_mvp/run_006/replay_report.json `
  --device cuda:0
```

规划 cuboid 即使位于主镜头外，也必须实际创建并通过 pose/size 回读，不能因不可见而省略。

## 10. 完成定义

只有同时满足以下条件才算完成：

1. run_006 四文件组合通过 A 级身份校验；
2. Isaac 机器人通过 B 级关节、限位和 FK 校验；
3. 专用场景通过 C 级 Base/World、对象和无额外碰撞体验收；
4. headless smoke 完成 35/35 帧；
5. GUI 实时回放通过 D 级，报告 `scene_consistent=true, passed=true`；
6. 非法文件、场景漂移、额外碰撞体和中途关闭均非零退出；
7. 文档与报告明确本轮是同规划场景 state replay，不是物理执行。

## 11. 后续迭代

1. 在规划产物中增加 `scene_snapshot.json`、`trajectory_sha256` 和显式 `target_name`；
2. 在同场景中增加 action/physics replay 与 tracking error；
3. 渲染 CuRobo collision spheres 调试 overlay；
4. 使用生产 world snapshot 重新规划和验收；
5. 增加严格 60 FPS 离屏采帧与视频编码。

## 12. 完成状态（2026-09-22）

本方案已完成。唯一命令为 `scripts/replay_curobo_trajectory_isaac.py`；最终 headless 报告为
`data/curobo_mvp/run_006/replay_headless_run_005.json`，GUI 正式报告为
`data/curobo_mvp/run_006/replay_report.json`。两者均为 `passed=true, scene_consistent=true`，
GUI 报告另为 `formal_acceptance_passed=true`。

A—D 级全部通过：四文件身份链与 world hash 一致；Isaac 关节、限位和 q0/qN project_tcp FK
一致；Base/World、两个 cuboid 的 pose/size 与碰撞 prim 清单一致；35 个源帧均恰好写入并显式
render 一次，无丢帧或重复 frame 0。详细数值、失败尝试和边界见
[分阶段验收](curobo_acceptance.md) 与 [施工进度](CuRobo施工进度_20260921.md)。

## 13. 后续同场景 physics replay 完成（2026-09-23）

第 11.2 节后续项已由独立命令 `scripts/replay_curobo_trajectory_physics_isaac.py` 完成。它沿用本文
全部输入身份与场景一致性硬门，frame 0 仅写初态，此后执行 q[1:] 位置目标；每个控制目标推进
两个 120 Hz 物理步，每个源帧显式 render 一次。最终 headless/GUI 报告分别为
`physics_replay_headless_run_002.json` 与 `physics_replay_report.json`，均通过；具体误差、实验门和
GUI 延迟见 [分阶段验收](curobo_acceptance.md)。本文仍是唯一可视化重播方案，已删除不要求
规划/仿真场景一致的旧基础 NPZ 方案。
