# 机器人数据生成管线

当前主链路固定为 **cuRobo 自由空间规划 + LegacyIK 接触段参考 + Isaac Sim 物理采集 + 独立审计**。
唯一的日常采集入口是 `scripts/collect_isaacsim.py`，默认 `physical` 模式。

先读 [主链路与实验链路](docs/主链路与实验链路.md)：包含调用顺序、脚本职责、数据流、配置入口、阅读顺序和实验合入规则。
完整文件迁移记录见 [机器可读清单](docs/pipeline_layout_20261015.json)。

```text
DataPipe/
├─ scripts/          主链路、回放及它们必需的共享模块（69 个 Python 文件）
│  ├─ native/        Isaac Sim 场景、机器人 IK 和 TCP 运动学
│  └─ pipelines/    规划、参考产物、物理执行、奖励及数据契约
├─ configs/          数据集、场景基线、相机、任务、控制和 cuRobo 配置
├─ testscripts/      实验、历史采集/复测、诊断、资产工具、训练模板及离线回归
├─ docs/             阅读说明、迁移清单与历史记录
├─ data/             已有采集数据
└─ outputs/          工具与离线校验产物
```

从本目录运行命令，使用已有 Isaac Python 环境：

```powershell
$isaac = "D:/Anaconda/envs/isaaclab/python.exe"

# 查看参数；不启动仿真
& $isaac scripts/collect_isaacsim.py --help

# 新任务：先规划参考，再 reset 后做物理采集
& $isaac scripts/collect_isaacsim.py --source-index 0 --destination-index 2 --gui --output <NEW_OUTPUT_DIR>

# 复用与当前 TCP 配置一致的参考，跳过规划
& $isaac scripts/collect_isaacsim.py --reference-run <REFERENCE_RUN> --gui --output <NEW_OUTPUT_DIR>

# 独立审计最终物理数据
& $isaac scripts/audit_physical_failure.py --run <RUN> --report <NEW_AUDIT_JSON>

# 查看物理回放参数；state/action 使用同一个入口
& $isaac scripts/replay_physical_trajectory_isaacsim.py --help

# 离线目录检查和已有回归
& $isaac testscripts/check_pipeline_layout.py
& $isaac -m pytest -q
```

命令中的 `<...>` 需替换为实际路径；新采集目录及其证据目录须不存在。
任务索引为 `0..4`，源和目标不能相同；也可使用 `--route-preset`。
`passed` 表示采集和数据审计通过，任务是否成功读取 `task_succeeded/outcome`，奖励能否训练读取 `reward_valid`。

`testscripts` 可调用主链路模块；主链路不依赖 `testscripts`。
旧 HDF5 入口现在是 `testscripts/collect_aubo_ik_dataset_isaac.py`，
稳定放置搜索是 `testscripts/search_stable_placement_isaacsim.py`，
历史整图实验是 `testscripts/collect_curobo_pick_place_v2_graph.py`。

配置入口集中在 [configs 说明](configs/README.md)：`dataset.py` 定义数据集契约，`baseline.py` 定义静态场景基线，`camera.py` 生成相机契约。

2026-10-15 首次目录清理及后续配置迁移的离线回归均为 **257 passed / 11 failed**，失败项一致。
首次清理保留的 72 个文件内容未变；后续将其中 3 个配置模块移入 `configs`，相关 Python 文件仅修改导入语句，配置值和执行逻辑不变。
本轮验证目录与依赖迁移，没有运行新的机器人仿真或宣称全部路线成功。详见主链路文档的验收记录。

安装依赖、旧 HDF5 接口及历史用法保存在 [历史管线说明](docs/历史管线说明_20261015.md)。
