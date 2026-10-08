# CuRobo 坐标契约（P1，尚未完成 P3 实测）

- 所有长度为 m，关节位置 rad、速度 rad/s、加速度 rad/s²、jerk rad/s³。
- 位姿布局 `xyz + wxyz`。Pose 不归一化输入四元数：非单位四元数直接拒绝。
- 所有规划 TCP 目标及轨迹 TCP 都表达于机器人 Base（`B`）。物体 source/place 位姿必须声明 frame 且两者一致。
- RobotConfig 保存实际 base/flange/TCP 链接名；其 flange_to_tcp 是 TCP 在 Flange 中的位姿。
- 不通过重排关节、切换 xyzw、裁剪限位或修正 TCP offset 自动掩盖配置不一致。

现有参考值位于 `configs/coordinate_frames.py`：F→T 平移 `(-0.17086, 0, 0.11305)`，旋转 `(sqrt(0.5), 0, sqrt(0.5), 0)`；现有六轴名称位于 `DataPipe/config.py`，为 Joint1…Joint5、Flange。这些是待验证参考，尚未写成新的可执行机器人 preset。

P3 必须分别测量当前 FK、Isaac FK、CuRobo FK，在相同关节姿态和 frame 下比较位置与旋转；还需核对关节序、限位、Base→World、Flange→TCP 和碰撞球。测试夹具中的零 offset 和虚构 joint0…joint5 仅供纯逻辑单测，绝不能用于采集。

## P3 首轮资产与 FK 实测（2026-09-21，尚未放行）

新增 `robot_adapter.py` 提供独立 URDF 串联链 FK、刚体变换合法性检查、六轴名称/限位对比及三方 FK 审计。生产模块不导入 Legacy IK；只有审计脚本读取当前 `forward_kinematics` 作为参考，不调用逆解。URDF FK 仅支持本任务所需的 fixed/revolute 链，不是碰撞或规划后端。

审计脚本：`scripts/audit_curobo_robot_isaac.py`。报告：`data/curobo_p3_audit_20260921_r2/report.json`，日志：`data/curobo_p3_audit_20260921_r2.log`。使用本机 isaaclab Python，14 个姿态（零位、默认位、seed=7 的 12 个 ±0.3 rad 扰动）。通过写入关节状态和更新运动学取得 Isaac FK，不在这些校准姿态间执行物理轨迹。请求与实测 q 最大差 5.82e-8 rad。

带夹爪 URDF：`D:/Project/S2R/Asset/AUBOE5_URDF/aubo_e5_with_claw.urdf`，SHA256 `b7c7c5cecd65fb730267e0f24f481979060f8d0d3b392a0880e89f9edeb959c5`。关节名称/顺序和位置限位与 Isaac 一致（限位差在 1e-6 rad 内）。以下问题仍阻止放行：

1. URDF velocity：前三轴 2.617994 rad/s，后三轴 3.141593 rad/s；Isaac 六轴实际均约 3.1400001 rad/s。不能静默覆写。
2. URDF 原生 `tool_center_point` 相对项目 TCP 有 0.01305059 m 位置差及 pi/2 旋转差。诊断对比明确用 URDF Flange 后乘项目 TCP offset，不把原生 TCP 的问题隐藏掉。
3. 实测 Base 四元数范数约 1.0000045769；场景配置使用截断值 `(0.70711, 0, 0, -0.70711)`，超过单位范数平方 1e-6 容差。审计记录原始值并将其计入硬阻塞；只有诊断矩阵显式归一化，契约本身不放宽。
4. 尚无真正的 CuRobo FK。报告 `missing_backends=['curobo']`，URDF 结果不冒充 CuRobo。

在统一项目 TCP、显式归一化实测四元数的诊断条件下：

| 对比 | 最大位置差 m | 最大旋转差 rad |
|---|---:|---:|
| 当前 FK / URDF | 3.2161e-7 | 0 |
| 当前 FK / Isaac | 4.9221e-7 | 1.1309e-6 |
| URDF / Isaac | 6.0721e-7 | 1.1309e-6 |

这些诊断通过 1e-5 m / 1e-3 rad 阈值，但总报告仍为 `passed=false, status=blocked`。没有修改外部资产、现有场景配置或 Legacy 行为。启动时的报告检查还需后续 planner/pipeline 接线；目前未生成可执行 RobotConfig preset、碰撞球或 CuRobo robot 配置。

P0 环境调查：本机 nvcc 为 CUDA 12.6，Torch 为 2.7.0+cu128，当前终端 PATH 未发现 cl。此处仅说明工具发现结果，不推断机器未安装 MSVC。cuRobo 尚未安装，未改动 isaaclab 依赖。官方[新版安装指南](https://nvlabs.github.io/curobo/latest/getting-started/installation.html)面向 Ubuntu；[v0.7.6 指南](https://nvlabs.github.io/curobo/v0.7.6/get_started/1_install_instructions.html)说明 Windows 为实验性支持，Python 3.11 未验证。版本锁定需结合本机真实构建和运行验收，不能仅选一个版本字符串。

## 2026-09-22 更新：派生资产与真实 CuRobo FK 验收通过

本节取代上一轮“缺少 CuRobo、三项资产差异”的阻塞状态。14 姿态报告 `data/curobo_p3_full_20260922/report.json` 为 `passed=true`，`contract_issues=[]`、`missing_backends=[]`。这是运动学与关节契约验收，不是完整规划/碰撞验收。

- `robot_asset.py` 和 `scripts/derive_curobo_robot_asset.py` 创建独立派生资产，拒绝覆盖目录或使用 hash 不匹配的源报告。
- 派生文件位于 `configs/curobo/assets/aubo_e5_project/`。原生 tool_center_point 保留，新增 Flange→project_tcp 固定变换。manifest 记录原始/派生 URDF SHA256、每个 mesh 的来源/hash 和逐关节速度转换。mesh 引用仍指向本工作区外部 Asset；这些不是可独立分发的自包含资产。
- 速度策略是 `min(原始URDF, Isaac实测)`；前三轴 2.617994 rad/s，后三轴约 3.14 rad/s。没有提高原始限位。
- CurriculumRL 的 `TcpDockingTrajectoryEnvCfg` 单独设置两台机器人及其 USD 父资产为精确的 -90° yaw 单位旋转，并给新环境 arm actuator 设置上述保守限位。旧场景配置/原始资产均未改写。
- `curobo_fk.py` 延迟加载实际 CuRobo 0.7.8，使用 GPU CudaRobotModel；拒绝版本及关节顺序不匹配。审计通过 `--with-curobo --tcp-link project_tcp` 启用，失败会记录原因，绝不 fallback 到 URDF。

最终最大差值：

| 对比 | 位置 m | 旋转 rad |
|---|---:|---:|
| 当前 FK / Isaac | 8.6317e-7 | 1.2519e-6 |
| 当前 FK / CuRobo | 4.8145e-7 | 4.1625e-4 |
| Isaac / CuRobo | 6.6468e-7 | 4.1625e-4 |

旋转误差使用 float32 FK 矩阵的 trace/acos 度量，靠近零时受浮点精度影响；此处如实记录实测值，验收阈值仍为 1e-3 rad，位置为 1e-5 m。新环境实测四元数已符合单位容差，报告不再含 quaternion issue。诊断转换仍显式记录在报告里。

下一步是 P4 世界模型、自碰撞球和 attach/detach。加速度/jerk 参数、完整碰撞模型和 planner 启动时的综合校验尚未完成，不能用本报告直接放行数据采集。
