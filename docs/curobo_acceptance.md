# CuRobo 分阶段验收

## P2：通用关节轨迹执行环境（2026-09-21 通过）

运行环境固定为本机 `D:/Anaconda/envs/isaaclab/python.exe`。实际使用 RTX 4060、Isaac Sim 5.1.0.0；无 CuRobo 依赖。本次验收只证明手工关节轨迹及空夹爪执行，不代表规划、抓取、碰撞安全或数据写盘通过。

### 实现位置

CurriculumRL 仓库 `source/CurriculumRL/CurriculumRL/tasks/tcp_docking/`：

- `joint_trajectory_action.py`：六轴绝对位置目标（rad）加二值夹爪命令，完整 batch 校验后缓存，在所有物理子步原样发送；没有 IK、增量缩放、插值和裁剪。拒绝 NaN/Inf、错误形状/dtype/device、越限目标和非二值夹爪命令。
- `trajectory_env_cfg.py`：独立 ManagerBasedEnv 配置及 Gym 注册适配类。复用动态场景，物理 120 Hz、默认控制 60 Hz。夹爪 open/closed preset 必须由调用方显式提供，并按全部环境的资产限位检查。
- `__init__.py`：追加 `CurriculumRL-TcpDocking-Trajectory-v0` 注册，旧注册保持不变。

DataPipe 仓库：`scripts/smoke_joint_trajectory_isaac.py`、`tests/curobo_bootstrap/test_joint_trajectory_action.py`。

### 调用契约

动作按 `Joint1, Joint2, Joint3, Joint4, Joint5, Flange, gripper` 排列，float32、设备与 articulation 一致。夹爪 -1 关闭、+1 打开。与 RL 环境不同，`step()` 返回 `(observations, extras)`，没有 reward/done，也没有自动 reset。观察为 `observations['state']['joint_position'/'joint_velocity']`，包括 articulation 所有关节，按资产顺序排列；调用方通过关节名称解析索引。

`reset(env_ids=...)` 后缓存对应环境当前关节位置和夹爪位置；raw action 的夹爪槽此时为 0，表示尚未收到离散命令，不是合法输入动作。下一次执行须提交完整七维动作。

本环境不提供任务成功/超时/碰撞中止判据；后续 executor 必须负责停止推进、保存 terminal observation，再显式 reset。其他继承 API（例如 reset_to）未纳入本次验收；使用前必须验证目标缓存同步。

### 实测

命令在 DataPipe 仓库根执行：

```powershell
& D:/Anaconda/envs/isaaclab/python.exe scripts/smoke_joint_trajectory_isaac.py --headless --device cuda:0 --output data/curobo_p2_smoke_20260921_r2/report.json
```

- 两个环境，预先固定的 360 条动作，361 次 observation；60 Hz 控制。
- Joint1 先平滑移动 +0.02 rad，保持并闭爪，再回原位并开爪。其他关节保持初始目标。
- 每步送入 articulation 的六轴目标与预定值完全一致，最大差 0 rad。
- 三个阶段末，最大关节跟踪误差均约 0.0022924 rad；夹爪最大误差分别约 0.0981、0.0935、0.0981 mm。阈值为关节 0.02 rad、夹爪 3 mm；这些是 smoke 阈值，不是正式任务验收阈值，也不是全轨迹误差峰值。
- NaN batch 被拒绝且控制器缓存保持；显式 reset 环境 0 后，环境 1 缓存不变。
- 报告：`data/curobo_p2_smoke_20260921_r2/report.json`；日志：`data/curobo_p2_smoke_20260921_r2.log`。
- 首次 smoke 未成功创建环境，无报告；定位后修复 Gym 参数适配，首次日志保留在 `data/curobo_p2_smoke_20260921.log`。Kit 关闭可能使失败进程也返回 0，因此验收必须检查报告中的 passed，不能只看退出码。

### 回归

- 新增动作测试 8 passed；CuRobo 目录总计 49 passed。
- DataPipe 全量：183 passed / 1 skipped。
- CurriculumRL 既有 `test_control_chain_split.py` 和 `test_sb3_action_contract.py`：8 passed。未声称整个相邻仓库的全部测试已运行。
- 本轮 Python 文件 Ruff 通过，修改的环境注册 diff 检查通过。

## 未完成的验收门

P0 仍缺 CuRobo 可运行版本锁定及新的 Legacy smoke/replay 基线。P3 三方 FK/机器人资产校验、P4 世界模型、P5—P12 后续阶段均未完成。P2 不依赖 CuRobo，故可独立验收，不解除这些后续验收门。

## P3：首轮对齐审计（2026-09-21，blocked）

14 姿态实测及关节配置差异已记录到 [坐标契约](curobo_frame_contract.md)。当前/Isaac/独立 URDF 的诊断 FK 几何一致，但 URDF TCP、速度限制、实测单位四元数尚有问题，且缺少真实 CuRobo FK，因此 P3 未通过。DataPipe 全量回归已增至 191 passed / 1 skipped。

## 2026-09-22：P3 运动学/关节契约通过

已解决上一节三项资产差异并接入本机真实 CuRobo 0.7.8。完整报告 `data/curobo_p3_full_20260922/report.json` 为 passed，14 姿态、无缺失后端、无契约差异。详情见 [坐标契约](curobo_frame_contract.md) 末节。P2 修正后复测也通过。全量回归 195 passed / 1 skipped。

复核命令：

```powershell
& D:/Anaconda/envs/isaaclab/python.exe scripts/audit_curobo_robot_isaac.py --headless --device cuda:0 --urdf configs/curobo/assets/aubo_e5_project/aubo_e5_project.urdf --tcp-link project_tcp --with-curobo --output data/curobo_p3_recheck/report.json
```

审计拒绝覆盖现有报告；复跑时使用新目录。只读机器人配置与 GPU FK 已通过，不代表 MotionGen、碰撞或数据采集通过；下一验收阶段是 P4。

## 2026-09-22：P4 世界状态与 attached object 基础链通过，生产几何待验收

新增 `DataPipe/pipelines/curobo_bootstrap/world_adapter.py`，提供不可变世界快照、稳定静态世界 hash、显式 WORLD/ATTACHED 状态机、TCP 局部附着球到 Base 坐标的变换，以及面向 CuRobo 0.7.8 的 primitive world 输入。attach 会在同一次状态转换中移除世界中的目标碰撞体并加入非空附着球；detach 会原子恢复目标碰撞体并清空附着球。非法的空附着、重复 attach/detach、部分状态和非有限球参数都会被拒绝。

真实 CuRobo CUDA primitive collision 冒烟在 RTX 4060 上通过。目标中心查询的有符号碰撞值按 `0.03 -> 0 -> 0.03 m` 变化，分别对应世界中存在、附着时移除、释放后恢复；静态世界 hash 在三种运行状态间保持不变。报告为 `data/curobo_p4_world_20260922/report.json`，其 `passed` 为 true。世界适配器测试 11 passed。

随后运行 `scripts/audit_curobo_world_isaac.py`，从生产 Isaac 单环境读取 C1/C2 工作站根、13 个托盘和样品瓶的实际世界边界，以第一台 AUBO 的实测 Base pose 转为 15 个固定障碍加 1 个目标的保守 oriented cuboid。CuRobo GPU 对全部 16 个 box 中心查询均为正，最小值约 0.23699 m；生产 world hash 为 `974100dc7f43339315452bebfc11899c356de9f14d1e18973bdfc7ffb4865889`。样品瓶世界 AABB 八角分别转换到 object frame 和实测 TCP frame，生成半径约 0.08925 m 的保守 TCP-frame attached sphere，并验证 attach 后目标 box 被移除。最终报告 `data/curobo_p4_production_world_20260922_r3/report.json` 为 passed；首次报告把 USD local bound 错当 object frame，r2 只修正 object frame，均保留为诊断而不作为最终证据。

生产几何当前仍是 profile 根的保守 AABB，并非逐碰撞 mesh 的精确近似；TCP-frame attached sphere 必须在实际 attach 时用当时实测的 object/TCP pose 重算。机器人自碰撞球、第二台机器人表示、真实机器人 world/self collision query、MotionGen 和真实抓取过程尚未验收。因此 P4 仍是部分通过，不能放行 P5 规划链或数据采集。

## 2026-09-22：固定关节目标 MotionGen MVP 通过

按 [CuRobo MVP 执行方案](CuRobo_MVP执行方案.md) 新增窄接口 `DataPipe/pipelines/curobo_bootstrap/planner.py` 和唯一命令 `scripts/plan_curobo_mvp.py`。它只规划 P4 `safe_q` 到 Joint1 `+0.1 rad` 的固定样例，使用两个远离机器人的静态 cuboid、primitive world collision、自碰撞、60 Hz 插值、最多 4 次尝试与 10 s 求解预算。位置和速度限制来自派生 URDF；加速度 `5 rad/s²`、jerk `50 rad/s³` 是本次显式实验配置，不是硬件认证值。

最终复核命令：

```powershell
& D:/Anaconda/envs/isaaclab/python.exe scripts/plan_curobo_mvp.py --device cuda:0 --output data/curobo_mvp/run_006
```

`data/curobo_mvp/run_006/report.json` 为 `passed=true`，真实后端为 CuRobo 0.7.8。保存并以 `allow_pickle=false` 重新读取 `trajectory.npz` 后，35 个点、0.5666667 s、时间间隔 1/60 s；最大单关节位移约 0.09999999 rad，起点误差约 `4.75e-9 rad`，终点误差约 `1.34e-8 rad`，最大速度约 `0.28058 rad/s`，最大加速度约 `1.88180 rad/s²`。碰撞结果形状为每采样点一个分数，全部采样点的位置/速度/加速度、自碰撞和世界碰撞检查通过。MotionGen 运行时回读的关节顺序和 position/velocity/acceleration/jerk 限制与请求值在 `1e-6` 容差内一致。MotionGen solve time 约 0.0340 s；planner 构造约 4.77 s，warmup 约 4.52 s，均不是性能门槛。

首次 `run_001` 已完成规划并写出轨迹，但报告序列化时因 CuRobo 的 tensor 型有效长度标记失败，没有 `report.json`，不能作为验收证据；修复为显式整数后 `run_002` 已通过。随后补齐 planner 构造耗时字段生成 `run_003`，增加碰撞查询必须返回 `[N]` 分数的门生成 `run_004`，加入 MotionGen 实际加载限位回读门生成 `run_005`；最后把配置构造也纳入失败报告保护区，`run_006` 是与最终代码一致的证据。没有覆盖或伪装失败尝试。

新增读回验收定向测试 9 passed，覆盖无 pickle NPZ、字段集合、错误时间步、非有限值、终点超差、碰撞误报、缺少碰撞后端及标量碰撞结果的假阳性。DataPipe 全量回归为 227 passed / 1 skipped，Ruff 与 `git diff --check` 通过。失败路径冒烟也确认缺失碰撞模型时退出码为 1、生成 `passed=false` 报告且不生成轨迹。本验收只证明固定合成世界中的离线关节空间规划与离散采样点检查；不证明连续扫掠净空、生产场景规划、Isaac 执行/跟踪、抓放、HDF5 或批量稳定性。

## 2026-09-22：NPZ 同规划场景可视化重播 MVP 通过

按 [场景一致可视化重播方案](CuRobo_NPZ场景一致可视化重播MVP施工方案.md) 新增纯逻辑
`DataPipe/pipelines/curobo_bootstrap/replay.py`、专用 Isaac 场景 `DataPipe/curobo_replay_scene.py`
和唯一命令 `scripts/replay_curobo_trajectory_isaac.py`。旧 HDF5 replay 未修改。

启动 Kit 前把 run_006 的 NPZ、规划报告、派生 URDF、碰撞模型四个 SHA256、报告状态、60 Hz
时间轴、MotionGen 关节顺序和 world hash 组成不可拆分的 A 级身份门。Isaac 场景只创建一台
AUBO、`probe`/`target` 两个规划碰撞 cuboid 和无碰撞灯光；cuboid 通过
`T_WO = T_WB @ T_BO` 生成并回读。碰撞模型的 `attachment_parent=project_tcp` 是 FK 末端事实源。

最终 headless 报告 `data/curobo_mvp/run_006/replay_headless_run_005.json` 与 GUI 正式报告
`data/curobo_mvp/run_006/replay_report.json` 均为 `passed=true, scene_consistent=true`；35 个源帧、
35 次状态写入、35 次显式 render，q/dq 写回误差和播放期间 Base/world pose 漂移均为 0。GUI
另为 `formal_acceptance_passed=true`。Base 位置回读误差约 `2.61e-8 m`；cuboid 最大位置误差
约 `2.65e-8 m`；q0/qN project_tcp FK 位置误差约 `6.29e-7/4.76e-7 m`，最大旋转误差约
`1.21e-4 rad`。没有额外 collision-enabled prim。

GUI wall-clock 约 `0.969 s`，有效帧率约 `35.09 FPS`，34 帧迟到，最大 lag 约 `371.33 ms`。
这些是 viewport/OS 调度性能证据，不是失败门；60 Hz 硬门是 NPZ 的 `1/60 s` 时间步、绝对
deadline、每源帧一次 render 和零丢帧，不宣称屏幕实际刷新或编码视频达到 60 FPS。

headless/GUI 报告 SHA256 分别为
`ed127470db45a7aeab4d299db3d7fd6b2431b7eaff901aaaf5dd520fc30629b8` 和
`c8bfa5571de92db7de12f739b7997f653ed26f36f88200862bf1a07c42b00378`。缺失 NPZ 的
`replay_invalid_input_report.json` 为 `passed=false`、退出码 1，且未启动 Kit。

失败尝试依次暴露并修复 PlanningWorld 父 Xform、Isaac Float 默认张量 dtype、位姿 finite 检查和
旧 `tool_center_point` 误用；最后一项的 13.05 mm/90° 偏差与历史审计一致，最终恢复冻结的
`project_tcp`，没有放宽 FK 阈值。新增 replay 测试 11 passed；全量回归 238 passed / 1 skipped。
Ruff、compileall 和 whitespace 检查通过，最终运行后无残留 Isaac/Kit 进程。本验收是同规划场景
state replay，不是 action/physics replay，不证明动力学跟踪、连续碰撞安全或生产世界通过。

## 2026-09-23：同规划场景 physics replay MVP 通过

新增独立入口 `scripts/replay_curobo_trajectory_physics_isaac.py`，继续复用已经通过 A—C 级门禁的
专用最小场景。frame 0 只写一次 q/dq 初态；随后仅发送 q[1:] 共 34 个位置目标。每个 60 Hz
控制点严格推进两个 120 Hz 物理步，并为每个源帧显式 render 一次。该入口不使用逐帧 state write
伪装物理跟踪，也不向执行器加入 NPZ dq 速度目标；dq 仅作为测量参考。

最终 headless 报告 `data/curobo_mvp/run_006/physics_replay_headless_run_002.json` 与 GUI 正式报告
`data/curobo_mvp/run_006/physics_replay_report.json` 均为 `passed=true, scene_consistent=true`；GUI
另为 `formal_acceptance_passed=true`。两次实测一致：34/34 目标、68 个物理步、35 次 render，
最大 q 误差 `0.02364073 rad`、q RMS `0.00693966 rad`、末帧最大 q 误差 `0.01129943 rad`；最大
dq 误差 `0.12625561 rad/s`、dq RMS `0.03259162 rad/s`。误差主要来自 Joint1。Base/world
pose 漂移均为 0。

本冻结样例使用的仿真验收门为全轨迹最大 q 误差 `<=0.03 rad`、末帧最大 q 误差
`<=0.015 rad`。这两个值由 run_001 的真实失败测量形成并在代码中明确标注，只是当前 AUBO
position actuator 与固定轨迹的 MVP 门，不是硬件精度声明。run_001 完整执行但按先验
`0.01/0.005 rad` 门判为 `passed=false`，原报告保留为阈值形成证据，没有改写。

GUI wall-clock 约 `1.219 s`，有效控制吞吐约 `27.89 Hz`，33 个目标迟到，最大 lag 约
`621.33 ms`。控制逻辑仍按每目标两个物理步完整推进且不丢帧；此结果不宣称 GUI 屏幕达到
60 FPS。后续若要严格实时显示，应做离屏采帧/编码或优化渲染，不能跳过物理步来追赶 deadline。
本验收仍不证明连续碰撞安全、生产世界规划、抓放、硬件跟踪或 HDF5 采集。

最终回归为 241 passed / 1 skipped；replay 定向测试 14 passed。Ruff、format check、compileall 与
whitespace 检查通过。headless/GUI 最终报告 SHA256 分别为
`0ab7d672862c6f92b950ed429d0566c328237f5bcc971f37bf586bf2df070cf7` 和
`8f4c8bc2223a958d6bfe384baacc3099ff0459edd6b677bd9bde501504a390e9`。

## 2026-09-23：CuRobo V2 0→1 混合 MVP 正式验收通过

按 [V2 敏捷施工方案](CuRobo第二版MVP敏捷施工方案.md) 和用户确认的边界，CuRobo **只规划两条
非接触段**；抓取接近、闭爪、抬升、放置接近和开爪使用 Legacy `AuboIKAction` 与夹爪控制器。
这不宣称抓取接触区零碰撞，也不把 Legacy 轨迹标成 CuRobo。规划入口在同一生产 Isaac 运行中
创建场景和同一个 warm MotionGen；首段从当次 reset 实测 q 出发，转运从 Legacy 抬升后的实测 q
出发，目标物按实测物体/TCP 位姿转换为 TCP-frame 保守附着球。首段和转运均重新读取并独立检查
导出采样点的 world/self 碰撞，分数为 0。Legacy 接触段另列诊断：最大 world 分数
`0.08888280`，self 为 0；这是预期接触范围的诊断，不是硬件安全放行。

正式复现命令（工作目录 `D:/Project/S2R/DataPipeline/DataPipe`）：

```powershell
& D:/Anaconda/envs/isaaclab/python.exe scripts/plan_curobo_pick_place_v2.py `
  --source-index 0 --destination-index 1 --device cuda:0 `
  --output data/curobo_pick_place_v2/0_to_1/run_003

& D:/Anaconda/envs/isaaclab/python.exe scripts/replay_curobo_pick_place_v2_isaac.py `
  --run data/curobo_pick_place_v2/0_to_1/run_003 --device cuda:0 --no-realtime
```

两个命令实测退出码均为 0。`run_003/plan_report.json.passed=true`；轨迹为 **758 帧**、
`12.6167 s`、60 Hz；首段 MotionGen 149 点、附着转运 120 点，均 `success=true` 且独立
限位/速度/加速度/离散 world/self 零碰撞门通过。第一段后端原始 status 为
`MotionGenStatus.GRAPH_FAIL`，但实际回退路径 `success=true`、导出验收通过，原状态保留在报告中。
执行分支五个 Legacy TCP 路标的 FK 残差均在 10 mm / 0.05 rad 内，实际物体末位置距 index1
约 `8.65 mm`。PICK/PLACE 事件帧为 199/757，逻辑附着只发生 `false→true→false`。

NPZ、场景快照和报告在写出后重新打开：`allow_pickle=False`、字段/shape、60 Hz、关节序、
阶段与来源、事件、夹爪、限位、接缝、hold 漂移、MotionGen 来源证据、Legacy FK/接触诊断、
世界/任务/资产 SHA256 均通过。生产场景 headless 与 GUI replay 报告均为 `passed=true`，
758 个源帧、758 次写入、758 次显式 render；GUI q 写回最大误差 0，place TCP 位置/姿态
误差约 `8.51 mm / 0.0384 rad`，目标物逻辑终态与 index1 位置误差约 `3.8e-8 m`。

正式文件 SHA256：`trajectory.npz` 为
`224140dbdf63205e287979a422fbe6d0b5861cac7e3238513cf8862024377b65`；
`scene_snapshot.json` 为 `a68c9438ebc1e7b25fc58297635c8de5e2b980f32078c95e3067bf7655d1b0ae`；
`plan_report.json` 为 `5c8555428c869bc48e584e7e6ee90323833651ca9b1dd7d8caa6ab69e4befac4`；
headless/GUI `replay_report` 分别为
`40dd7a055de0e713dd32bbc1df669af2bfe45bf0bb55f69612c4385cfb6bbc5e` /
`9b01313b29f3ce893ccf6d23d1d9acc71370e772f3bbe61b3962588979f24cd0`。

失败和中间尝试保留：`integrated_probe_001/002` 与 `run_001/002` 分别用于打通同场景流程、
增加 reset 起点门和完善报告；它们不取代最终 `run_003`。最终全量测试 267 passed / 1 skipped，
Ruff check/format 与 `git diff --check` 通过（只有既有 LF/CRLF 提示）。V1 `run_006` 未修改。
本 MVP 的 CuRobo 段和 replay 都是离散采样/state replay 证明；转运物体为逻辑附着，
不证明连续扫掠净空、接触力、抓取稳定性、硬件执行安全或 action/physics tracking。
