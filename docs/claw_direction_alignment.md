# 放置朝向的物理轴与控制器轴

放置要求为机器人资产 `/Root/AUBO_E5/ClawTool_C/ClawTool_C` 子 prim 的
`+Y_C` 对齐 `R_WG · (-Y_G)`，全程保持法兰 `-X_F` 朝世界 `-Z_W`。
这里 C 是子 prim 局部系，T 是 IK 控制器的计算 TCP 系，不能将二者混用。

当前 `AUBO_E5/AUBO_E5_Withclaw.usd` 的实际变换为：

- `q_FC = (0.5, 0.5, 0.5, 0.5)`，所以 `+Y_C → +Z_F`。
- `q_FT = (sqrt(0.5), 0, sqrt(0.5), 0)`，所以 `-X_T → +Z_F`。
- 因此 `TCP_DIRECTION_AXIS_T = (-1, 0, 0)`，表示所需物理轴在 IK TCP 系中的表达。

控制目标和到位判定均使用这个轴。对世界系的实测方向：

```text
d_actual_W = R_WB R_BT (-X_T)
           = R_WF R_FC (+Y_C)
d_target_W = R_WG (-Y_G)
```

旧代码直接控制 `+Y_T`，它对应 `+Y_F`，与所需的 `+Z_F` 相差 90 度。
更换轴表达不更改 q_FT、不改变 TCP 位置标定，也不关闭法兰朝下约束。
下降及最终位置确认阶段同时检查方向，防止高位对齐后发生漂移仍松爪。

磁盘 USD 中子 prim 的局部变换为单位阵，所以子 `+Y_C` 对应父 prim 的
`+Y`，并非父 `+Z`。父 `+Z` 当前对应 `+X_F`，在法兰朝下时朝世界上方。
本次以子 prim 的 `+Y` 为唯一物理目标，不重命名或修改资产层级。

新采集数据记录 `tcp_heading_axis_local`；回放按每条 episode 的轴解释 heading。
缺少该字段的旧数据仍按旧 IK TCP `+Y` 解释，避免旧动作被静默旋转 90 度。

验证位于 `tests/test_claw_direction.py`：包含错误物理轴禁止下降/松爪的反例、
四种目标状态的真实 heading 几何求解，以及实际 USD 子 prim 的标定检查。
后两项需要 CurriculumRL/torch 和 pxr；普通 Python 缺少运行库时会跳过，
应在带这些库的运行环境中另行执行。几何验证不等同于 Isaac 动力学抓放验证。
