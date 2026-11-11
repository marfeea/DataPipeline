# 管线配置

本目录保存主链路的配置与静态契约，不依赖 `scripts` 或 `testscripts` 的运行时实现。

| 入口 | 内容 |
| --- | --- |
| `dataset.py` | schema 版本、动作维度/单位/缩放、字段规格、`DatasetConfig` 和 `CameraSensorSpec`。 |
| `baseline.py` | 静态 IK 场景基线、目标状态名称和来源描述。 |
| `camera.py` | 将项目视觉配置转换成三路相机契约及数据集配置。 |
| `task.py` | 五状态任务与抓取/放置 TCP 目标。 |
| `coordinate_frames.py` | 坐标系、TCP 与法兰约定。 |
| `assets.py`、`scene.py` | 资产路径、机器人 prim 契约与场景配置。 |
| `aubo_ik_control.py` 等控制配置 | IK、关节和差分控制参数。 |
| `vision_diagnostics.py`、`render.py` | 观测相机成像模型/位姿与视口设置。 |
| `curobo/` | 机器人、世界、规划、验收参数、路线预设和相关资产。 |
| `training.py`、`rewards.py`、`curriculum*.py` | 保留的训练、任务奖励与课程配置。 |

```python
from configs.dataset import DatasetConfig
from configs.baseline import STATIC_IK_SCENE_BASELINE
from configs.camera import dataset_config_with_project_diagnostic_cameras
```

物理采集主入口仍是 `scripts/collect_isaacsim.py`。
配置的使用顺序与实验合入规则见 [主链路说明](../docs/主链路与实验链路.md)。
