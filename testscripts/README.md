# 实验与历史脚本

主采集入口是 `scripts/collect_isaacsim.py`。本目录集中存放实验、历史采集/诊断/复测、训练模板和离线回归。
职责分类、阅读顺序及合入标准见 [主链路与实验链路](../docs/主链路与实验链路.md)。
106 个迁移文件的旧路径、新路径与哈希见 [完整清单](../docs/pipeline_layout_20261015.json)。

从项目根目录运行：

```powershell
python testscripts/check_pipeline_layout.py
python -m pytest -q
python testscripts/evaluate_pick_place_rewards.py --help
```

依赖方向为 `testscripts -> scripts/configs`，主链路不依赖实验目录。
新功能先放独立实验子目录，使用独立入口、配置和新输出目录，取得同条件验证证据后再合入主链路。
`curobo_bootstrap/test_*.py` 及根目录的 `test_*.py` 是离线回归，pytest 不会自动执行其他采集/诊断脚本。

`tasks/` 与各 RL 框架目录保留原训练模板和 YAML，需要相应外部环境。
本次只验证离线语法、依赖和现有回归，没有启动训练模板或机器人仿真。
