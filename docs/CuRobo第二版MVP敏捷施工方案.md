# CuRobo 第二版 MVP 敏捷施工方案

日期：2026-09-23。

修订（2026-09-23）：按 Legacy IK 抓放实现，将抓取/放置接触段交回 Legacy IK 控制器；
CuRobo MVP 只规划非接触自由空间段。以下正文以此修订为准，不再要求五段均由 MotionGen 生成。

## 1. 唯一目标

在现有 CuRobo MVP、Legacy IK 任务几何和 Isaac replay 基础上，生成并重播一条固定的
`index0 -> index1` Pick & Place 轨迹。

本轮交付只在以下条件同时满足时完成：

1. 从生产场景 reset 状态出发，使用 Legacy IK 已有坐标与姿态规范求出全部任务关节目标；
2. 两段非接触运动由真实 CuRobo MotionGen 生成，接近、闭爪、抬升、放置接触段复用 Legacy IK
   逐步控制，并组成完整四阶段执行记录；
3. 轨迹保存后重新读取，在与 Legacy IK 一致的生产场景中完成一次全帧 replay；
4. 规划报告和 replay 报告均为 `passed=true`；
5. 产物明确标识 `source_index=0`、`destination_index=1`，且所有验收来自真实 CuRobo/Isaac 运行。

本轮不接 HDF5、相机或批量采集。首先证明一条轨迹可以规划、保存、读回和重播。

## 2. 固定范围与技术决策

### 2.1 复用 Legacy IK 的任务事实源

不重新定义 index、TCP、四元数或工作站坐标：

- `configs/task.py::TARGET_STATES` 是 index0/index1 目标物、preposition 和 grasp TCP 的唯一事实源；
- 复用 Legacy collector 的 `E -> W -> B` 坐标转换；
- 复用 `flange_down_tcp_target_pose()`、TCP heading、tool-forward axis 和现有 wxyz 约定；
- 复用 `_pick_place_positions_b()` 的安全高度规则，得到 lift、place-high 和 final-place 位姿；
- 复用 Legacy IK 后端，把上述 TCP 位姿求解为六轴关节目标；CuRobo V2 不新增第二套 IK 几何。

Legacy IK 负责五个关节目标、接近/离开接触区的逐步控制和诊断；CuRobo 只负责标记为
`curobo` 的非接触段。报告和逐帧产物必须记录每段来源，不能把 Legacy IK 轨迹包装成 CuRobo 结果。

### 2.2 四阶段状态机

顶层状态固定为：

```text
MOVE_TO_PICK -> PICK -> MOVE_TO_PLACE -> PLACE -> COMPLETE
```

各阶段语义如下：

| 阶段 | V2 MVP 行为 | 完成条件 |
|---|---|---|
| `MOVE_TO_PICK` | CuRobo `q_start -> q_pre_pick`；Legacy IK 从预停靠位逐步接近 `q_pick` | 到达 index0 抓取 TCP |
| `PICK` | 机械臂保持，按 Legacy 控制闭爪并等待到位；记录唯一 `PICK` 事件 | 闭爪指令及停留帧完整 |
| `MOVE_TO_PLACE` | Legacy IK 抬升至 `q_lift`；CuRobo `q_lift -> q_place_high`；Legacy IK 逐步放置至 `q_place` | 到达 index1 放置 TCP |
| `PLACE` | 机械臂保持，按 Legacy 控制开爪并等待到位；记录唯一 `PLACE` 事件 | 开爪指令及停留帧完整 |

四个顶层阶段是任务状态。固定执行路由为两条 CuRobo 非接触 query、三条 Legacy IK 接触/离开
段和两次夹爪事件；实现需逐帧保存运动来源，不增加新的顶层状态。

`PICK`/`PLACE` 在本轮是轨迹事件，不代表真实接触、夹持力、抓取稳定性或放置稳定性已验证。
replay 可以按事件切换夹爪目标和目标物的逻辑附着状态，但不得将逻辑附着表述为物理抓取验收。

### 2.3 碰撞策略

- 沿用 CuRobo 当前机器人、生产静态世界、自碰撞和目标物表示；
- 仅对两条 CuRobo 非接触 query 开启已有 world/self 碰撞检查；任一 query 规划失败或其任一导出
  采样点发生碰撞，整条 run 失败；
- Legacy IK 接触段沿用生产 Isaac 控制与实测到位门，保留碰撞诊断，但不冒充 CuRobo 无碰撞轨迹；
  当前已知抓取位姿的下指与目标/托盘 11 相交，必须在报告中显式标注，不能宣称全程无碰撞；
- 失败 run 写 `passed=false` 报告，不进入 golden replay，不进入未来数据集；
- 不通过关闭 CuRobo 非接触段碰撞、删除生产障碍物、裁剪失败帧或把 Legacy 轨迹冒称 CuRobo
  使结果通过。接触段只限 Isaac 仿真 MVP，不作为硬件执行安全证明。

本轮不解决连续扫掠净空、最小安全距离和更精细的 mesh 碰撞建模。验收只声明当前模型下的
两段 MotionGen 成功及其离散导出采样点无碰撞；不声明 Legacy 接触段无碰撞。

### 2.4 生产场景规范

规划和 replay 均对齐 Legacy IK 使用的生产规范：

- 同一台 AUBO、同一机器人 Base pose、同一关节名称和顺序；
- 同一 `TARGET_STATES[0]` 与 `TARGET_STATES[1]`；
- 同一环境原点、TCP 标定、wxyz 四元数和 `E/W/B` 变换链；
- 同一工作站、托盘及目标物初态；
- 规划报告冻结任务配置、机器人资产和场景快照 hash，replay 在播放前重新核对。

不得继续使用 V1 的两个远端合成 cuboid 作为 V2 正式验收场景。

## 3. 产物契约

每次运行使用全新目录，例如：

```text
data/curobo_pick_place_v2/0_to_1/run_001/
  trajectory.npz
  scene_snapshot.json
  plan_report.json
  replay_report.json
```

### 3.1 `trajectory.npz`

使用 `allow_pickle=False` 可重新读取的数据类型，至少包含：

| 字段 | 形状 | 说明 |
|---|---:|---|
| `time_s` | `[N]` | 从 0 开始、60 Hz、严格递增 |
| `joint_position` | `[N,6]` | 六轴位置，rad |
| `joint_velocity` | `[N,6]` | 六轴速度，rad/s |
| `joint_acceleration` | `[N,6]` | 六轴加速度，rad/s2 |
| `joint_names` | `[6]` | 与机器人、CuRobo、Isaac 完全同序 |
| `stage_id` | `[N]` | 四阶段枚举；不使用隐式时间范围推断 |
| `segment_source` | `[N]` | `curobo/legacy_ik/gripper_event`，不能混淆轨迹来源 |
| `event_code` | `[N]` | `NONE/PICK/PLACE`，每个事件只出现一次 |
| `gripper_target` | `[N,2]` | replay 目标；不作为抓取成功证据 |
| `object_attached` | `[N]` | replay 的逻辑附着状态 |

拼接 CuRobo 子轨迹与 Legacy IK 实测控制记录时删除相邻重复端点，统一重建全局 60 Hz 时间轴。
Legacy 段的速度/加速度须按记录来源定义并标注，不能标为 MotionGen 插值结果。`PICK` 和
`PLACE` 停留段向机械臂发送保持命令，记录实际 q/dq 漂移并校验阈值，不伪造零速度测量。

### 3.2 `scene_snapshot.json`

保存本次实际读取的：

- source/destination index、名称和完整 `TargetState`；
- `T_WE`、`T_WB` 及必要的坐标约定；
- 机器人资产、TCP 标定和六轴契约；
- 生产静态障碍物及目标物初态；
- world hash、任务配置 hash 和相关资产 SHA256。

replay 只能读取该快照恢复或校验场景，不能在回放时重新猜测 index 几何。

### 3.3 报告

`plan_report.json` 至少记录：

- 输入身份、CuRobo 版本、设备和随机种子；
- 六个任务关节目标及每个目标的 Legacy IK residual/有效性；
- 两条 MotionGen query 的起终点、状态、尝试次数、求解耗时和点数；三条 Legacy IK 段的控制步数、
  到位误差和实测状态；
- 拼接轨迹的阶段/来源区间、事件帧、限位，以及仅针对 CuRobo 非接触段的离散碰撞检查；
- NPZ、scene snapshot、URDF 和 collision model 的 SHA256；
- `passed`、失败阶段和原始错误。

`replay_report.json` 至少记录场景身份检查、源帧/写入帧/render 次数、关节写回误差、TCP
端点误差、阶段与事件顺序、逻辑附着转换、是否完整播放以及最终 `passed`。

## 4. 计划中的最小代码结构

优先扩展窄接口，不改动 V1 golden sample 的语义：

- `DataPipe/pipelines/curobo_bootstrap/task_targets.py`：从生产 Isaac 状态和 Legacy 几何生成、校验任务 TCP/关节目标；
- `DataPipe/pipelines/curobo_bootstrap/hybrid_route.py`：冻结两段 CuRobo、三段 Legacy IK 与两次夹爪事件的所有权；
- `DataPipe/pipelines/curobo_bootstrap/pick_place_v2.py`：四阶段执行、逐步 Legacy IK 控制、两段 MotionGen 调用、实测记录拼接和事件生成；
- `DataPipe/pipelines/curobo_bootstrap/pick_place_artifact.py`：NPZ/快照/报告的保存与读回验收；
- `scripts/plan_curobo_pick_place_v2.py`：固定 index0->index1 的唯一规划入口；
- `scripts/replay_curobo_pick_place_v2_isaac.py`：生产场景全帧 replay 入口；
- `tests/curobo_bootstrap/test_pick_place_v2.py`：纯逻辑与误报防护测试。

可复用的 V1 `CuroboMvpPlanner` 应提取为“单段关节规划”能力或以组合方式调用；不得复制一套
MotionGen 初始化、轨迹校验和碰撞查询实现。Legacy collector 中的纯几何函数应移动到共享模块后
由旧入口和 V2 同时调用，避免从带 CLI/Isaac 启动副作用的脚本中导入。

## 5. 敏捷迭代计划

每个 Sprint 都必须形成可运行增量、自动测试和明确停点。只有当前 Sprint 通过后才进入下一项，
不提前建设 HDF5、批量采集或通用任务编译器。

### Sprint 0：冻结故事、输入和验收场景

用户故事：作为开发者，我能用唯一配置精确表示 index0->index1，而不会混用 V1 合成世界或旧
坐标约定。

工作：

1. 冻结 source=0、destination=1、60 Hz、固定 seed 和生产 scene ID；
2. 把 Legacy collector 的目标位姿计算提取为无副作用共享接口；
3. 生成 `scene_snapshot.json` 草案并校验 hash 稳定性；
4. 写配置、坐标系、index 越界和 source=destination 的定向测试。

停点：同一 reset 重复提取的任务位姿和 scene hash 必须一致；index0/index1 的事实源只能来自
`TARGET_STATES`。

### Sprint 1：Legacy IK 任务关节目标

用户故事：作为规划器，我能获得每个 CuRobo 子段需要的、可审计的六轴关节终点。

工作：

1. 在生产 Isaac reset 后读取 `q_start`；
2. 计算 `pre-pick`、`pick`、`lift`、`place-high`、`place` 五个 TCP pose；
3. 按实际运动顺序以上一解为 seed 调用 Legacy IK；
4. 对每个结果做 finite、关节限位、FK position/orientation residual 和关节顺序检查；
5. 输出只包含目标诊断的 dry-run 报告。

停点：五个目标全部有效，FK 回算满足沿用的 Legacy IK 容差；任一目标无解即失败，不进入
MotionGen，不静默替换姿态或 waypoint。

### Sprint 2：四阶段混合执行

用户故事：作为操作者，我能得到一条带明确阶段和运动来源的完整 0->1 关节轨迹。

工作：

1. 复用同一个 warm MotionGen 实例规划 `q_start -> q_pre_pick` 与 `q_lift -> q_place_high` 两段；
2. 复用 Legacy IK 的逐步 TCP 控制完成接近 `pick`、抓后抬升、放置接近，闭爪/开爪时机械臂保持；
3. 每段起点使用上一段实际终点，尤其在 Legacy 抬升后以实测 q 作为 CuRobo 转运起点；检查
   接缝误差及实测 TCP 到位条件；
4. 拼接并重建时间轴、stage、segment_source、gripper 和逻辑附着字段；
5. 对完整记录重跑 finite/限位/时间轴门；速度/加速度按来源核对，仅对 CuRobo 非接触采样点
   做 world/self 零碰撞硬门，并单列 Legacy 接触段的碰撞诊断。

停点：内存中的完整执行记录通过对应来源的全部门，状态严格按四阶段单向转换，任何子段失败则
整条失败；不能以 Legacy 段绕过非接触 CuRobo 失败。

### Sprint 3：冻结文件并读回验收

用户故事：作为验收者，我不依赖进程内对象，也能证明轨迹及其生产场景身份正确。

工作：

1. 原子写出 NPZ、scene snapshot 和 plan report；
2. 使用 `allow_pickle=False` 重新读取 NPZ；
3. 校验 hash、字段集合、shape、finite、60 Hz、阶段、来源、事件、接缝、起终点和相应来源的碰撞结果；
4. 为错误 hash、错序 stage、重复事件、错误时间步、端点跳变和部分文件补失败测试。

停点：`plan_report.json.passed=true`，且删除内存结果后仍能仅凭三个文件通过读回验收。

### Sprint 4：生产场景 replay

用户故事：作为操作者，我能在 Legacy IK 同规范生产场景中看到整条 0->1 轨迹完整播放。

工作：

1. Kit 启动前完成 NPZ/报告/快照/资产身份门；
2. 启动与 Legacy IK 一致的单环境生产场景，核对 Base、关节、TCP、目标物和障碍物；
3. 首版采用确定性 state replay，逐帧写 q/dq 和 gripper target，每源帧显式 render 一次；
4. `PICK` 后将目标物按冻结的 TCP 相对变换做逻辑附着，`PLACE` 后放到 index1 并解除；
5. 每帧检查关节写回、场景静态不变量、阶段/事件以及目标物逻辑状态；
6. 先完成 headless no-realtime smoke，再完成一次 GUI 正式 replay。

停点：`source_frames = written_frames = render_calls`，没有跳帧或重复事件，最终机械臂到达
index1 放置关节目标，目标物逻辑终态等于 `TARGET_STATES[1]`，报告 `passed=true`。

该 replay 只证明轨迹和场景契约可重放，不证明 action/physics tracking 或真实抓放。

### Sprint 5：真实 GPU 验收与收尾

用户故事：作为项目负责人，我能用两条固定命令复现规划与 replay 结果。

预定命令：

```powershell
# 工作目录：D:/Project/S2R/DataPipeline/DataPipe
& D:/Anaconda/envs/isaaclab/python.exe scripts/plan_curobo_pick_place_v2.py `
  --source-index 0 --destination-index 1 --device cuda:0 `
  --output data/curobo_pick_place_v2/0_to_1/run_001

& D:/Anaconda/envs/isaaclab/python.exe scripts/replay_curobo_pick_place_v2_isaac.py `
  --run data/curobo_pick_place_v2/0_to_1/run_001 --device cuda:0
```

工作：运行定向测试、真实规划、headless replay 和 GUI replay；将实际命令、版本、产物 hash、
失败尝试和最终结果追加到 `docs/curobo_acceptance.md`。首次正式通过即停止本轮，不顺手扩展数据
采集或批量能力。

## 6. 验收标准（Definition of Done）

以下各项必须全部通过：

### A. 任务目标

- source/destination 精确为 0/1，名称与 `TARGET_STATES` 一致；
- 五个任务 TCP pose 与 Legacy 共享接口输出一致；
- 五个 Legacy IK 关节目标 finite、限位内，FK residual 达标；
- `q_start` 来自同一次生产场景 reset，不使用 V1 `SAFE_Q` 代替。

### B. 四阶段轨迹

- 顶层阶段严格为 `MOVE_TO_PICK, PICK, MOVE_TO_PLACE, PLACE`；
- 两条非接触 CuRobo 子段均由真实 MotionGen success 结果组成；接近、抬升和放置接触段均明确
  标为 Legacy IK 实测控制；
- 相邻子段接缝连续，PICK/PLACE 停留段保持指令及实测漂移可审计；
- `PICK`、`PLACE` 事件各且仅各一次，逻辑附着状态只允许
  `false -> true -> false`；
- 全部有效采样点满足本次有限值、关节限位和来源对应的速度/加速度门；CuRobo 非接触采样点
  满足离散 world/self 零碰撞门，Legacy 接触采样点不声明无碰撞。

### C. 文件与身份

- NPZ、snapshot 和 plan report 均存在、可独立读回且 hash 相互匹配；
- 时间从 0 开始、严格递增、固定 `1/60 s`；
- 所有数组长度一致、数值有限、关节名称与顺序正确；
- V1 产物仍能通过既有测试，V2 不改写 run_006 golden sample。

### D. Replay

- replay 使用生产场景规范和同一 scene snapshot，不使用 V1 合成场景；
- 播放前机器人、Base、TCP、目标、静态世界和资产身份全部一致；
- 全部源帧恰好写入和 render 一次；
- 关节写回与阶段端点在显式阈值内；
- 最终状态为机械臂位于 index1 放置位姿、目标物逻辑位于 index1；
- `plan_report.passed=true`、`replay_report.passed=true`，两个命令退出码均为 0。

单元测试、Legacy IK 可达性、V1 单关节规划或仅能加载 NPZ 均不能单独替代上述验收。

## 7. 失败策略

- 任一阶段失败立即停止后续阶段，并尽可能写 `passed=false`、失败 stage 和原始异常；
- 失败目录不复用、不覆盖，后续尝试使用新的 run 编号；
- 不输出或标记为 golden，不进入未来 HDF5 数据；
- 不自动放宽 FK、关节端点、时间步、碰撞或 replay 阈值；
- 修复必须针对已复现问题，随后从最近可重复的 Sprint 验收点重跑。

## 8. 本轮明确不做

- 不验证连续扫掠碰撞和最小净空；
- 不验证真实夹爪接触、抓取力、物体稳定性或放置稳定性；
- 不做 physics/action replay 精度验收；
- 不接 HDF5、相机、reward、训练 schema 或 dataset validator；
- 不做多 seed、多 index 边、批量成功率、性能优化或失败恢复；
- 不做第二台机器人协同规划；
- 不重构成通用任务编译平台。

这些项目只有在本方案生成并重播一条正式 0->1 轨迹后，才进入下一轮 backlog。
