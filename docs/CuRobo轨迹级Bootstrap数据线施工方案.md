# CuRobo 轨迹级 Bootstrap 数据线施工方案

> 2026-09-22 施工顺序调整：当前按 [CuRobo MVP 执行方案](CuRobo_MVP执行方案.md) 推进，先交付“一条命令生成轨迹并读回验收”。本文保留为完整数据线的后续路线参考，P0—P12 不再作为首条轨迹的串行前置条件。

## 1. 建设目标

在不改变现有 Legacy IK 数采链行为的前提下，新增一条独立的 CuRobo 轨迹级专家数据生成线：

```text
场景配置 + 任务配置 + 机械臂配置
                ↓
           EpisodeSpec
                ↓
        TrajectoryProgram
                ↓
        CuRobo 分段轨迹规划
                ↓
      拼接、重采样与轨迹预检
                ↓
        Isaac Sim 关节轨迹执行
                ↓
      轨迹评估、任务评估与留存
                ↓
      当前数据规范兼容视图 + 原生轨迹字段
```

最终保留两条互不干扰的数据生成链：

```text
Legacy IK：状态机 → 手写 TCP 路径 → 数值 IK → Isaac Sim → HDF5
CuRobo：任务编译 → CuRobo 完整轨迹 → Isaac Sim → 评估 → HDF5
```

## 2. 核心原则

1. 不修改现有 Legacy IK 数采链的运行语义。
2. CuRobo 新线不依赖现有 `sequence.py`、`trajectory.py` 和 AUBO 数值 IK。
3. CuRobo 主体位于 DataPipe；CurriculumRL 仅提供通用关节轨迹执行环境。
4. 第一版先完成完整轨迹规划，再交给 Isaac Sim 开环执行，不启用在线重规划。
5. 运行时反馈只用于抓取/放置验证和安全中止，不用于改变剩余轨迹。
6. 数据继续遵循 `T+1` observation、`T` action/reward 的时序契约。
7. Legacy IK 与 CuRobo 数据使用独立输出目录、manifest 和 shard。
8. 规划失败、执行失败和任务失败均需结构化记录，不得静默丢弃。
9. 常规 action replay 直接执行已保存的关节轨迹，不重新调用 CuRobo。

## 3. 推荐目录

### 3.1 DataPipe

```text
DataPipe/
├─ configs/
│  └─ curobo/
│     ├─ __init__.py
│     ├─ planner.py
│     ├─ robot.py
│     ├─ world.py
│     ├─ bootstrap.py
│     └─ acceptance.py
│
├─ DataPipe/
│  └─ pipelines/
│     └─ curobo_bootstrap/
│        ├─ __init__.py
│        ├─ contracts.py
│        ├─ episode_sampler.py
│        ├─ task_compiler.py
│        ├─ robot_adapter.py
│        ├─ world_adapter.py
│        ├─ planner.py
│        ├─ trajectory.py
│        ├─ executor.py
│        ├─ evaluator.py
│        ├─ record_adapter.py
│        └─ pipeline.py
│
├─ scripts/
│  ├─ collect_curobo_bootstrap.py
│  ├─ replay_curobo_execution.py
│  ├─ replay_curobo_planning.py
│  └─ audit_curobo_bootstrap.py
│
├─ tests/
│  ├─ curobo_bootstrap/
│  └─ isaac/
│     ├─ smoke_curobo_single_segment.py
│     ├─ smoke_curobo_pick_place.py
│     └─ smoke_curobo_recording.py
│
└─ docs/
   ├─ CuRobo轨迹级Bootstrap数据线施工方案.md
   ├─ curobo_data_contract.md
   ├─ curobo_frame_contract.md
   └─ curobo_acceptance.md
```

### 3.2 CurriculumRL 最小新增

```text
CurriculumRL/source/CurriculumRL/CurriculumRL/tasks/tcp_docking/
├─ joint_trajectory_action.py
└─ trajectory_env_cfg.py
```

新增环境建议命名为：

```text
CurriculumRL-TcpDocking-Trajectory-v0
```

其动作只包含绝对关节目标和夹爪命令：

```text
[q1, q2, q3, q4, q5, q6, gripper]
```

该环境不得调用 CuRobo、数值 IK 或 TCP 增量缩放。

## 4. 模块职责与依赖

```text
configs
  ↓
contracts
  ↓
episode_sampler
  ↓
task_compiler
  ↓
robot_adapter + world_adapter
  ↓
planner
  ↓
trajectory
  ↓
executor
  ↓
evaluator
  ↓
record_adapter
  ↓
EpisodeBuilder / HDF5 writer / validator
```

依赖约束：

- `contracts.py` 不依赖 CuRobo、Isaac、HDF5 或 CurriculumRL。
- `task_compiler.py` 不访问 Isaac 运行时。
- `planner.py` 不写数据、不控制 Isaac。
- `executor.py` 不依赖 HDF5 schema。
- `evaluator.py` 只计算指标，不修改轨迹或执行结果。
- `record_adapter.py` 不调用 CuRobo 或 Isaac。

## 5. 核心数据对象

### 5.1 EpisodeSpec

描述一条可重建的采集请求，至少包含：

- episode ID 和随机种子；
- source/place state；
- 初始关节状态；
- 抓取和放置 TCP 位姿；
- lift、pregrasp、preplace 参数；
- robot/task/scene/world 配置 hash。

### 5.2 TrajectoryProgram

由 `task_compiler.py` 从 `EpisodeSpec` 编译得到：

```text
PREGRASP
APPROACH_GRASP
CLOSE_GRIPPER
VERIFY_GRASP
LIFT
TRANSFER
PREPLACE
PLACE
OPEN_GRIPPER
VERIFY_PLACE
RETREAT
```

其中：

- 机械臂运动表示为 `MotionSegment`；
- 夹爪、dwell、attach/detach 和验证表示为 `DiscreteEvent`；
- 可继续沿用当前稳定的 `semantic_stage_id`，但不调用现有状态机。

### 5.3 PlannedTrajectory

统一重采样后的完整轨迹至少包含：

```text
time_s                  [T]
joint_position          [T, 6]
joint_velocity          [T, 6]
joint_acceleration      [T, 6]
planned_tcp_pose_b      [T, 7]
gripper_command         [T]
segment_index           [T]
segment_sample_index    [T]
semantic_stage_id       [T]
event_flags             [T]
```

## 6. 规划与执行流程

### 6.1 机器人契约检查

启动时必须校验：

- Isaac 与 CuRobo 关节名称及顺序；
- 关节位置、速度限制；
- Base、Flange 和 TCP frame；
- Flange-to-TCP offset；
- 四元数 `wxyz/xyzw` 约定；
- 米和弧度单位；
- 当前 FK、Isaac FK 与 CuRobo FK 的一致性。

任何一项不一致均应拒绝启动，不能静默修正。

### 6.2 世界模型

第一版优先支持确定性场景几何：

- 工作台及固定设备；
- 目标物体和其他障碍物；
- 机器人自碰撞球；
- 抓取后的 attached object。

世界适配器必须提供稳定的 `world_hash`，并明确处理 attach/detach。

### 6.3 分段规划

```text
q0 → PREGRASP → q1
q1 → APPROACH_GRASP → q2
attach object
q2 → LIFT → q3
q3 → TRANSFER → q4
q4 → PREPLACE/PLACE → q5
detach object
q5 → RETREAT → q6
```

后一段以前一段终点为起点。任一必要段规划失败时，不进入 Isaac 执行，也不自动 fallback 到 Legacy IK。

### 6.4 拼接与预检

CuRobo 多段结果需要经过：

1. 关节顺序统一；
2. 重复边界点删除；
3. 夹爪 dwell 插入；
4. 段间位置和速度连续性检查；
5. 重采样到统一 `control_hz`；
6. planned TCP FK 计算；
7. 关节限位、速度、加速度、jerk 和碰撞检查；
8. 当前动作规范范围检查。

超出单步动作范围时应增加采样密度或重新规划，不能通过裁剪改变专家轨迹。

### 6.5 Isaac 执行

执行器按时间索引逐步发送：

```text
planned q[t] + gripper[t]
            ↓
JointTrajectoryActionTerm
            ↓
Isaac physics
            ↓
actual q/dq/TCP/object/camera/contact
```

运行时只允许：

- 正常推进；
- 抓取/放置验证；
- 碰撞、超差和环境终局中止。

第一版不允许根据实际误差修改剩余轨迹。

## 7. 当前数据规范兼容

### 7.1 保持不变

- observation 为 `T+1`；
- action/reward 为 `T`；
- `o0-a0-r0-o1-...-oT`；
- terminal observation 在 reset 前写入；
- `is_first/is_last/is_terminal`；
- 三路相机同步；
- Base/World/Object 坐标语义；
- manifest、checksum、分片与原子写入；
- state replay 和 action replay。

### 7.2 v4 动作兼容视图

通过相邻规划关节目标的 FK 生成：

```text
[dx, dy, dz, 0, 0, 0, tcp_heading_target, gripper]
```

- `dx/dy/dz` 来自相邻 planned TCP 位移；
- legacy 旋转增量保持为零；
- heading 来自 planned TCP 姿态；
- `action_applied` 来自实际发送关节目标的 FK；
- 重采样必须保证动作自然落入规范范围，禁止事后裁剪。

### 7.3 CuRobo 原生字段

新增：

```text
planner/
├─ time_from_start
├─ joint_position
├─ joint_velocity
├─ joint_acceleration
├─ tcp_pose_b
├─ segment_index
├─ segment_sample_index
├─ semantic_stage_id
├─ collision_distance
└─ event_flags
```

旧 `ik_*` 字段如需兼容现有 reader，只能作为明确标注的兼容映射。metadata 至少记录：

```text
controller_family=curobo_trajectory
motion_generation_backend=curobo
action_source=fk_projection_of_planned_joint_trajectory
ik_field_semantics=curobo_compatibility_projection
online_replanning=false
fallback_used=false
planner_version
planner_config_hash
robot_config_hash
world_config_hash
task_config_hash
```

CuRobo episode 不得与 Legacy IK episode 混写到同一 shard。

## 8. 评估与数据留存

### 8.1 规划质量

- 规划是否成功；
- 规划耗时和尝试次数；
- 最小碰撞距离；
- 轨迹长度与时长；
- 关节限位余量；
- 速度、加速度、jerk；
- 段边界连续性。

### 8.2 执行质量

- q tracking RMSE/max；
- TCP tracking RMSE/max；
- 轨迹完成率；
- 非目标碰撞；
- 夹爪跟踪误差；
- 执行超时和安全中止原因。

### 8.3 任务质量

- 抓取是否成功；
- 搬运过程中是否掉落；
- 放置位置和姿态误差；
- 释放后是否稳定；
- retreat 是否完成。

### 8.4 留存分级

```text
accepted/
    schema、规划、执行和任务评估全部通过

quarantine/
    规划成功，但执行或任务失败；保留完整 episode

rejected/
    规划或 preflight 失败；保留请求和失败诊断
```

建议输出结构：

```text
data/curobo_bootstrap/<run_id>/
├─ run.json
├─ config_hashes.json
├─ accepted/
├─ quarantine/
├─ rejected/
├─ manifests/
├─ audits/
└─ cache/
```

## 9. 回放策略

### Execution replay

读取保存的 joint trajectory 和 gripper command，直接交给 Isaac 执行，不调用 CuRobo。这是数据可重现性的主要验收方式。

### Planning replay

根据 `EpisodeSpec + seed + config hash + planner version` 重新调用 CuRobo，用于检查规划配置和版本变化，不作为日常 action replay。

## 10. 施工阶段与验收门

| 阶段 | 主要工作 | 完成标志 |
|---|---|---|
| P0 | 冻结 Legacy 基线、锁定依赖版本 | 旧测试、smoke、validator 和 replay 通过 |
| P1 | 数据契约和配置 | 中立对象、序列化、hash 测试通过 |
| P2 | 通用关节轨迹环境 | 手工关节轨迹和夹爪 smoke 通过 |
| P3 | 机器人与坐标对齐 | Isaac/当前/CuRobo FK 对齐 |
| P4 | CuRobo 世界模型 | 障碍物和 attach/detach 对齐 |
| P5 | Episode 采样与任务编译 | 稳定生成 TrajectoryProgram |
| P6 | CuRobo 分段规划 | 完整 pick-place 各段规划成功 |
| P7 | 拼接、重采样和 preflight | 完整轨迹满足执行约束 |
| P8 | Isaac 执行与采集 | planned/sent/actual 和相机完整 |
| P9 | 数据规范适配 | 当前公共 validator 和新 validator 通过 |
| P10 | 评估与分级留存 | accepted/quarantine/rejected 可审计 |
| P11 | execution/planning replay | 两类回放报告可生成 |
| P12 | 批量稳定性 | 1、10、100 episode 分级验收通过 |

## 11. 推荐提交批次

1. `curobo-contracts-config`
2. `generic-joint-trajectory-env`
3. `curobo-robot-world-adapters`
4. `curobo-task-planning`
5. `curobo-trajectory-processing`
6. `curobo-isaac-execution`
7. `curobo-v4-storage`
8. `curobo-evaluation-replay`
9. `curobo-batch-acceptance`

## 12. 最终完成标准

- Legacy IK 全部回归测试继续通过；
- CuRobo 新线不调用现有状态机、手写路径和数值 IK；
- 机器人关节、FK、TCP 和世界坐标契约通过启动校验；
- 完整 pick-place 可提前生成固定关节轨迹；
- Isaac 执行层不二次改变规划轨迹；
- 数据满足 `T+1/T`、相机同步、终局捕获和当前公共 schema；
- CuRobo 原生轨迹、配置、版本、seed 和 hash 完整保存；
- action replay 不需要重新运行 CuRobo；
- accepted、quarantine、rejected 分类明确；
- 批量采集支持原子写入、断点续采和审计。
