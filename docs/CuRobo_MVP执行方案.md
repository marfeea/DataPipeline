# CuRobo MVP 执行方案

日期：2026-09-22。本文为当前施工入口，原 P0—P12 方案保留为后续路线参考，不再作为本轮串行前置门。

## 1. 唯一目标

一条命令调用真实 CuRobo MotionGen，生成一条非零六轴关节轨迹，保存文件，重新读取并自动验收。

本轮交付为“可运行命令 + trajectory.npz + passed=true 的 report.json”。下文保留施工前拟定的范围和门槛；截至 2026-09-22 已按本方案完成，最终状态见第 6 节。

## 2. 固定范围与复用基础

- 环境：`D:/Anaconda/envs/isaaclab/python.exe`、本机 CuRobo 0.7.8、`cuda:0`。
- 场景：一台 AUBO E5、固定起终关节位置、简单静态 cuboid 测试世界、空夹爪、无附着物。
- 本轮只离线规划，不启动 Isaac 执行，不生成训练数据。
- 复用派生 URDF、碰撞球、`collision_model.py`、`world_adapter.py`、`curobo_collision.py`；资产和接口未变时复用已有审计证据。

已核实的通过报告：

| 能力 | 报告（相对 DataPipe 仓库根） |
|---|---|
| 三方 FK 对齐 | `data/curobo_p3_full_20260922/report.json` |
| 手工关节执行 | `data/curobo_p2_corrected_20260922/report.json` |
| 合成场景 robot/world/self/attachment GPU 检查 | `data/curobo_p4_robot_collision_20260922/report.json` |

施工记录末节尚未包含第三份报告，不应根据旧状态重复建设碰撞基础链。这些结果均不代表 MotionGen 或生产场景路径已验收。

## 3. 三步施工

### 第一步：完成一次真实规划

新增 `DataPipe/pipelines/curobo_bootstrap/planner.py`，只负责 MotionGen 初始化、固定关节目标规划和结果提取。

1. 复用 `curobo_robot_dict()`，在规划侧补齐已锁定版本需要的 cspace、retract、加速度和 jerk 参数；不改变既有碰撞适配器语义。
2. 起点使用现有碰撞报告的 `safe_q`；终点首选 Joint1 增加 `0.1 rad`，其余关节不变。先检查两端限位与碰撞；若无效，只调整固定样例并记录原因，不增加随机采样器。
3. 固定简单测试世界，开启自碰撞与世界碰撞检查，调用 `MotionGen.plan_single_js()`。
4. 提取有效长度的插值轨迹，目标频率 60 Hz，核对实际采样间隔，排除预分配缓冲区中的无效点。
5. 复用现有配置的最多 4 次尝试、10 秒求解预算；初始化耗时另记，本轮不设性能门槛。

位置、速度限制复用派生资产；加速度和 jerk 采用明确的实验配置并记录，不能宣称为硬件认证值。不以关闭碰撞、裁剪目标或切换 Legacy IK 的方式使样例通过。

### 第二步：形成唯一命令入口

新增 `scripts/plan_curobo_mvp.py`，直接组织固定样例、规划、保存、读回验收与报告。暂不新增通用 pipeline、任务编译器或执行框架。

预定调用方式（待实现）：

```powershell
# 工作目录：D:/Project/S2R/DataPipeline/DataPipe
& D:/Anaconda/envs/isaaclab/python.exe scripts/plan_curobo_mvp.py --device cuda:0 --output data/curobo_mvp/run_001
```

输出使用新目录；复跑换目录名。成功只要求两个文件：

- `trajectory.npz`：`time_s [N]`、`joint_position [N,6]`、`joint_velocity [N,6]`、`joint_acceleration [N,6]`、`joint_names [6]`，使用无需 pickle 即可读取的数据类型。
- `report.json`：passed、失败阶段/原因、Python/CuRobo 版本、起终点、机器人/世界标识、实际限制与阈值、采样点数/时长、初始化/规划耗时、逐项检查结果及轨迹文件路径。复用已有资产 hash，不新增 hash 框架。

这里的 N 个点是从 t=0 开始的规划状态，不是 N 条执行动作；不强行套用现有 PlannedTrajectory 的 command-end 时间轴或抓放阶段字段。下一轮接执行时明确处理首个初始状态点。

### 第三步：重新读取文件并验收

下表针对重新加载的文件计算，不能只检查内存中的规划结果。

| 检查 | 通过标准 |
|---|---|
| 后端 | 真实 MotionGen 返回 success |
| 数据 | N >= 2，字段长度一致，六轴顺序正确，全部数值有限 |
| 非零运动 | 终点相对起点的最大单关节位移至少 0.01 rad |
| 时间轴 | 从零开始严格递增，间隔 1/60 s，允许显式浮点容差 |
| 起点 | 相对请求起点的最大单关节误差 <= 1e-4 rad |
| 终点 | 相对请求终点的最大单关节误差 <= 0.01 rad |
| 运动限制 | 导出 q/dq/ddq 满足本次配置的位置/速度/加速度限制，数值容差显式记录 |
| 碰撞 | 用同一机器人和世界配置检查全部有效采样点，自碰撞与世界碰撞检查通过 |
| 完成 | 文件可读，以上全部通过，report.json 中 passed=true，退出码为 0 |

以上误差阈值是首轮拟定标准，并非已有实测结论；失败先定位原因，不通过放宽阈值掩盖问题。现有碰撞查询的零值不表示正安全距离，本轮结果仅证明该模型的采样点检查通过，不包含连续扫掠或最小净空证明。

失败返回非零退出码；能够写报告时保存 passed=false、失败阶段和原始原因。报告缺失、轨迹缺失或仅退出码为 0 均不能验收。无需建立 rejected/quarantine 留存体系。

## 4. 验证、停点与问题处理

- 为读回验收补少量定向测试，覆盖有效文件及错误时间轴、非有限值、终点超差等误报风险。
- 用唯一入口完成一次真实 GPU 规划与读回验收；单元测试不能替代这个结果。
- 检查修改文件的格式与差异，仅在共享接口被修改时扩大相关回归范围。
- 在现有施工记录末尾补充实际命令、产物路径、结果、遇到的问题和下一步。
- 一次真实端到端通过即结束本轮并交付；不追加抽象、批量测试或完整生产几何审计作为完成条件。

如果 MotionGen 初始化或运行失败，优先修复阻断该命令的依赖、配置或 API 问题。FK/碰撞成功和新增审计模块均不能替代首条规划轨迹。新增防御逻辑须对应当前输入边界或已复现问题。

## 5. 后续迭代顺序

1. 读取保存轨迹，接现有关节执行环境；确保规划世界与仿真场景一致，记录实际关节轨迹和跟踪误差。
2. 固定一个 source/place，完成抓取、搬运、释放，再补真实接触与附着处理。
3. 接 HDF5、相机、T+1/T 与公共 validator，随后做 10 条、100 条稳定性验收。

随机采样、通用任务编译、生产世界精化、v4 动作投影、分级留存、planning replay 和断点续采均留在后续待办。

## 6. 完成状态（2026-09-22）

MVP 已完成。唯一命令为 `scripts/plan_curobo_mvp.py`，最终真实 GPU 证据位于 `data/curobo_mvp/run_006/`：`trajectory.npz` 和 `report.json` 均存在，后者为 `passed=true`。实现、实测数值、失败尝试及边界见 [分阶段验收](curobo_acceptance.md) 和 [施工进度](CuRobo施工进度_20260921.md)。

最终实现还增加了两项防误报门：碰撞后端必须为全部 N 个采样点分别返回 world/self 分数；MotionGen CUDA 模型实际加载的关节顺序和 position/velocity/acceleration/jerk 限制必须与请求值一致。它们没有扩展本方案的功能范围。
