# CuRobo 数据契约 v1（P1）

实现：`DataPipe/pipelines/curobo_bootstrap/contracts.py`。该模块仅依赖标准库；配置位于 `configs/curobo/`。Legacy collector、writer、schema 和状态机未修改。

## 请求、程序与轨迹

- `EpisodeSpec` 保存 episode ID、63-bit 非负 seed、六轴初始 q、source/place 物体位姿、Base 中抓取/放置 TCP 位姿、距离及 dwell 参数，以及 robot/task/scene/world/planner 五类 SHA256。
- 当前 source/place state 只冻结为刚体位姿；速度、动态物体状态与执行期测量尚未实现，不得把配置位姿当成真实反馈。
- `TrajectoryProgram` 固定为施工方案中的 11 个阶段；运动与离散事件是不同对象。序列化显式保存 `kind` 和版本，未知种类/版本拒绝加载。
- 抓取验证事件包含 ATTACH；放置验证事件包含 DETACH。它们表示离线世界拓扑和运行期验证的意图，尚无 CuRobo attach/detach 或物理约束实现。后续 planner 与 executor 必须分别解释，不能依靠标志假装完成抓取。
- `PlannedTrajectory` 存储 T 个命令：六轴 q/dq/ddq、planned TCP、夹爪、程序步索引、步内样本索引、语义阶段和事件。夹爪 `-1` 关闭、`+1` 打开，与 Legacy 数据符号一致。
- `segment_index` 指向 `program.steps`，包含 dwell 程序步；步内索引从零连续计数。阶段跳过/事件只出现一次等跨对象语义留待 P7 检查。
- `semantic_stage_id` 当前只约束为非负整数，尚未冻结 Legacy 兼容编号映射。不得直接使用枚举顺序代替公共 schema 阶段 ID。
- 碰撞距离缺省为 `None`，表示尚未计算；不得用零冒充已通过碰撞检查。
- `FailureDiagnostic` 保存 episode、phase、code、message 和可选失败段；持久化分类目录尚未实现。

## 时间轴

`time_s[i] = (i+1)/control_hz`，表示第 i 条关节命令对应控制区间的结束时刻。初始 q0 在 `EpisodeSpec`，初始 observation 在未来执行器中先捕获。执行顺序为 o0 → 发送 q[0] → 推进一个控制周期 → o1，因此 T 命令必须配 T+1 observation。不得把首条命令作为初始 observation 或再执行一次 q0。

目前验证的是轨迹结构与均匀时间轴，尚未验证数值导数、限位、jerk、FK 或碰撞；不能仅凭对象构造成功进入 Isaac 执行。

## 序列化和 hash

EpisodeSpec、TrajectoryProgram、PlannedTrajectory 提供 JSON round-trip。使用排序键、紧凑 UTF-8 JSON 和 SHA256；NaN/Inf 拒绝序列化。对 dataclass 配置调用 `config_hash(dataclasses.asdict(config))`。世界 cuboid 按名称排序后哈希，输入列表顺序不改变世界标识。

hash 是精确内容标识，不做浮点容差比较或四元数符号等价归并。机器人模型文件、碰撞球、配置文件的字节摘要还需由后续 adapter 纳入配置；当前只有几何配置 hash 不能证明资产一致。

## 配置边界

- PlannerConfig 要求显式版本字符串和 CUDA device，禁止 online_replanning/fallback。版本存在性与实际运行版本匹配尚待 P0/P6 校验。
- RobotConfig 要求显式六轴名称、位置/速度/加速度/jerk 限制、frame 和 TCP offset，不内置未经测量的机器人参数。
- WorldConfig 当前只支持 Base 坐标中的 cuboid；这不是完整的 CuRobo world adapter。
- AcceptanceConfig 阈值必须显式给出；当前尚无评估器。
- BootstrapConfig 限制 run ID 为单个安全路径组件，输出根末级目录必须为 curobo_bootstrap。目录创建、manifest/shard 隔离及已有数据族检查属于 P9/P10；目前尚无 collector。

包级限制：现有 `DataPipe/__init__.py` 会提前导入 Legacy 模块。当前仅保证 contracts.py 自身标准库依赖，不保证经 `import DataPipe...` 的整个包初始化完全脱离 h5py/numpy。为避免改变 Legacy API，本批未重构包初始化。
