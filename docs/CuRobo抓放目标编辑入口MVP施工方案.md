# CuRobo 抓放目标编辑入口 MVP 施工方案

日期：2026-09-24  
状态：MVP 已实施并验收；五点 CuRobo 有向图测试继续进行。

## 1. 目标与交付边界

让当前 CuRobo V2 抓放链像 LegacyIK 采集入口一样，由操作者指定**抓取目标状态**和**放置目标状态**，并使目标提取、探针、规划、产物校验和重播使用同一对目标。

本轮“编辑目标”指从 `configs/task.py::TARGET_STATES` 选择源和目标索引。目标状态的位置、姿态、预停靠点和抓取 TCP 点继续由 `configs/task.py` 统一定义；本轮不新增第二份目标几何配置或图形界面。当前 CuRobo V2 链交付的是单条混合抓放轨迹及重播证据，尚不是 LegacyIK 的 HDF5 批量数据集采集入口。

MVP 交付物：

1. 正式命令行入口能选择任意**合法且不同**的源、目标索引，不再只接受 `0→1`。
2. 每次运行的目标、世界和探针证据按所选目标对生成或显式传入，并完成身份核对；不能把旧 `0→1` 证据用于新目标对。
3. 规划产物与重播报告记录实际目标索引、名称和任务 hash；读回校验按任务快照核对。
4. README 提供从目标选择到轨迹重播的可执行示例。

“可选择”不等于保证五个状态间所有有向目标对都能规划成功。IK、碰撞或 MotionGen 失败应产生明确失败报告，不得标为通过。

## 2. 当前实现与卡点

| 环节 | 当前实现 | 施工关注点 |
|---|---|---|
| LegacyIK 入口 | `scripts/collect_aubo_ik_dataset_isaac.py` 使用 `--target-state-index` 选抓取源、`--place-state-index` 选放置目标 | 与其索引语义对齐 |
| 目标事实源 | `configs/task.py::TARGET_STATES` 定义五个状态；`DataPipe/pipelines/curobo_bootstrap/task_targets.py::PickPlaceV2Config` 已验证索引范围和源/目标不同 | 复用已有定义、坐标变换和五点生成 |
| 目标提取 | `scripts/dry_run_curobo_pick_place_v2_targets_isaac.py` 当前构造默认 `PickPlaceV2Config()`，即 `0→1` | 加入所选索引并写入报告 |
| 正式规划 | `scripts/plan_curobo_pick_place_v2.py` 虽要求 `--source-index`、`--destination-index`，却显式拒绝非 `0→1`；默认探针路径位于 `0_to_1`，转运 goal 还读取固定探针文件 | 去除固定目标对及固定证据路径的耦合 |
| 集成执行 | `scripts/probe_curobo_v2_legacy_contact_isaac.py` 从目标、首段和世界报告继续执行 | 校验三份输入对应同一任务和场景 |
| 产物构建 | `scripts/build_curobo_pick_place_v2_artifact.py` 校验 `0→1`，报告也写死索引 | 从任务快照填写并交叉校验 |
| 读回校验 | `DataPipe/pipelines/curobo_bootstrap/pick_place_artifact.py::verify_hybrid_artifact` 拒绝非 `0→1` | 校验合法索引、任务 hash 和产物内部一致性 |
| 重播 | `scripts/replay_curobo_pick_place_v2_isaac.py` 从产物快照读取源和目标 | 确认重播场景及报告与所选目标一致 |

现有 `0→1` 报告属于该目标对的历史证据；改变目标对后，至少目标 IK 和两段规划探针需要重新取得对应证据。世界证据若可复用，也必须核对世界 hash、机器人资产和场景身份，不能仅凭文件路径推定兼容。

## 3. MVP 施工步骤

### 步骤 A：定义一次运行的任务身份

- 复用 `PickPlaceV2Config` 的 `source_index`、`destination_index` 和现有 `task_snapshot()`；以其中的索引、状态名称、任务 hash、坐标变换及五个 waypoint 为运行事实。
- 给目标 dry-run 入口暴露源/目标索引。报告保存实际配置、完整任务快照和 Legacy IK 结果。
- 统一参数命名：CuRobo 正式入口保留现有 `--source-index`、`--destination-index`；README 明确对应 LegacyIK 的 `--target-state-index`、`--place-state-index`。
- 入口启动时校验索引类型、范围及源/目标不同；帮助信息或错误信息应能让操作者查到 `TARGET_STATES` 的可用名称。

### 步骤 B：让证据跟随目标对

- 调整正式入口，不再读取写死在 `data/curobo_pick_place_v2/0_to_1` 下的默认首段、目标、世界和转运探针。可采用由本次 `--output` 派生的独立证据目录，并由入口顺序生成所需证据；若保留显式报告参数，必须对全部报告做任务身份校验。
- 生成首段和转运段 goal 时使用本次目标快照与实测 reset，而不是固定 `0→1` 探针的关节值。
- 集成执行开始前核对目标报告、探针报告、世界报告的 `passed` 状态、目标对、任务 hash、起点关节及世界 hash。缺失身份字段的旧报告不能用于新目标对。
- 对失败阶段保留 `passed=false`、输入目标对和错误原因；不进入最终产物构建。

### 步骤 C：贯通产物和重播

- 产物构建从本次任务快照读取源/目标索引、名称及任务 hash，删除 `0→1` 常量判断和报告中的固定索引值；保留 60 Hz 等当前轨迹契约所需的独立检查。
- `verify_hybrid_artifact()` 校验快照 hash、索引范围、源/目标不同，以及报告、轨迹、快照中的任务身份一致；不再把 `0→1` 当作合法性的条件。
- 重播继续消费冻结产物，并核对当前生产场景与快照一致；重播报告记录实际目标对。现有 CuRobo 非接触段与 LegacyIK 接触段的来源标记和验收门保持有效。

### 步骤 D：文档与回归

- 在 `README.md` 写明可用状态、参数映射、为新目标对生成证据及规划/重播的命令顺序，输出使用新的目录，避免覆盖历史 `0→1` 证据。
- 为参数校验、非默认目标快照、跨目标对报告拒绝、产物身份一致性添加针对性测试。
- 先完成无需 Isaac 的单元测试，再对一个非 `0→1` 目标对运行真实 Isaac/CuRobo 链；保留原 `0→1` 回归。

## 4. 验收标准

| 编号 | 预期结果 | 判定方法 |
|---|---|---|
| A1 | `0→1` 原流程保持可用 | 重新生成新目录下的目标、规划和重播报告；各阶段 `passed=true`，读回校验通过 |
| A2 | 非 `0→1` 目标对真正进入任务链 | 选择一对不同索引，检查目标报告的源/目标名称、五个 waypoint、任务 hash 与该对 `TARGET_STATES` 一致；集成和产物报告采用同一身份 |
| A3 | 对可行目标对完成端到端验证 | 非默认目标对若通过 IK 和规划门，保存轨迹、重新读入并在生产 Isaac 场景重播；规划、产物校验和重播报告均为 `passed=true` |
| A4 | 错误输入不会混入产物 | 越界、源目标相同、错配的目标/探针报告或旧 `0→1` 任务 hash 均明确失败，且不生成标为通过的最终产物 |
| A5 | 失败可诊断 | 合法但不可达或碰撞失败的目标对，报告保留所选索引、失败阶段和原因；不将失败归为“入口不支持” |

若所选非默认目标对真实规划失败，先依据报告判断是目标本身不可行还是入口/证据传播缺陷。只有后者属于本轮必须修复的问题；前者应如实记录，并换一对可行目标完成 A3。若没有任何非默认目标对可通过，则 A3 是阻塞项，不能宣称端到端验收完成。

## 5. 后续迭代

在单条目标对轨迹及重播通过后，再考虑：批量遍历目标对、CuRobo 轨迹写入 HDF5 数据集、按目标名称而非索引选择、独立任务配置文件或图形编辑界面。这些均不作为本轮 MVP 的完成条件。

## 6. 给后续施工对话的起点

先检查当前工作树及上述入口、报告 schema 和测试；仓库内已有未提交的 CuRobo V2 文件，施工时保留现有改动与历史证据。优先完成目标身份从 CLI 到产物的贯通，再运行真实 Isaac/CuRobo 验收，不要仅删除 `0→1` 判断而继续复用固定探针。


## 7. 实施与验收记录（2026-09-24）

- A1：在新目录 data/curobo_pick_place_graph_20260924/0_to_1 重新生成目标、世界、CuRobo 两段、Isaac 同次抓放与混合轨迹；run_report.json、plan_report.json、replay_report.json 均通过，轨迹 737 采样点，重播生成 replay_visualization。
- A2/A3：非默认 1→0 在 data/curobo_pick_place_graph_20260924/1_to_0_retry 完成相同闭环；任务索引、状态名称、task hash 与证据一致，轨迹 734 采样点，读回和生产场景重播通过，生成三相机可视化帧。
- A4：15 项目标身份与产物单元测试通过；相同索引 CLI 在创建目录前拒绝；用 1→0 世界报告和 0→1 目标报告调用真实分支筛选脚本，因 world source_index differs from selected task 退出且没有输出文件。
- A5：每个候选的 first_probe_*、contact_probe_*、place_high_branches_*、transfer_probe_* 和集成报告保留失败阶段与原因；入口尝试其他候选，整条路线失败时 run_report.json 保留身份、阶段和日志。1→0 与 0→1 均在第四个抓取候选成功，证明单分支失败不会被误判为路线不可达。

后续 20 条有向边由 scripts/collect_curobo_pick_place_v2_graph.py 逐条运行，汇总写入 data/curobo_pick_place_graph_20260924/graph_summary.json。未通过的边仅表示在记录的候选与规划预算内未采集成功。


## 8. 后续五点有向图结果

CuRobo 正式入口在 Isaac Lab cuda:0 中测试了 5 个目标状态间全部 20 条有向路线。最终汇总见 data/curobo_pick_place_graph_20260924/RESULTS.md 和 graph_summary.json：6 条完成混合轨迹采集、读回、生产场景重播及三相机可视化；14 条在当前 IK/碰撞/MotionGen/混合轨迹验收预算内未采集成功。0→2 的首次放置分支因实测加速度超限，第二分支通过，完整保留两次证据。未采集成功的路线没有被宣称为物理上绝对不可达。


## 9. 采样瓶接触与托盘硬门禁（2026-09-27）

预抓取与 CuRobo 首段规划只从机器人—世界碰撞查询中排除本次目标采样瓶网格 target；support_tray_*、玻璃和工作台等其他网格继续参与碰撞，自碰撞继续为硬门禁。世界和预抓取报告同时记录含 target 与排除 target 的碰撞分数、受保护网格名。抓取后附着的瓶子继续作为转运碰撞模型；Legacy 接触段的产物验收单独检查机器人—托盘，允许瓶子放置时与托盘接触，并记录逐阶段分数与碰撞托盘名。

状态 2 作为抓取源的 2→0、2→1、2→3 已用新规则重新运行预抓取筛选：每条路线 8 个 IK 候选、0 个通过，自碰撞分数仍为正；允许接触采样瓶并未解除此阻塞。证据位于 data/curobo_pick_place_graph_20260924 各路线证据目录中的 pre_pick_branches_target_contact.json。

旧版 6 条“已采集并重播”轨迹用新机器人—托盘门禁逐帧复核后均未通过；旧产物与可视化保留为历史证据，不再视为新策略合格轨迹。具体托盘名、分数与报告链接见 data/curobo_pick_place_graph_20260924/RESULTS.md。新的五点整图测试输出至 data/curobo_pick_place_graph_tray_strict_20260927，与历史证据分开；20/20 条均已在 Isaac Lab 测试，当前 0 条通过完整轨迹与重播门禁。9 条停在目标 IK、3 条停在预抓取自碰撞、8 条被机器人—托盘碰撞门禁拒绝。逐路线证据和托盘名见该目录的 RESULTS.md 与 graph_summary.json。
