# 已验收实测轨迹的 GUI 重播

目的：按用户要求，将 `legacy_timed_lift_20261021_verified` 三条实测轨迹重播，每 12 个控制步输出三路原生相机图像。

- 源轨迹：每条 411 步，60 Hz；同时核对 NPZ 与 trace 的关节位置、速度、关节顺序。
- 方式：可见 IsaacSim GUI，按记录恢复机械臂、手指及物体实际姿态；不重新规划，不运行物理步。未记录的手指及物体速度置零，仅用于显示。
- 图像：384×384，三个项目原生相机，严格捕获第 12、24……408 步，每条 34 组、102 张。不会额外补拍阶段关键帧。
- 证据：原始输入 SHA256、每步状态回读误差、原生相机渲染参考和外参、步号/时间/阶段索引、图片完整性核验和可视化总览。
- 轨迹来源索引保存在 `data/test/trajectorys/<run-id>/0_to_1/`；日志、图片及总览保存在 `data/test/trajdata/<run-id>/0_to_1/`。
- 这是历史实测状态的可视化重播；不作为新的动力学测试结果。生产脚本和源数据不修改。

运行：`D:\Anaconda\envs\isaaclab\python.exe testscripts/replay_timed_lift_20261021/replay.py --backend isaacsim`
