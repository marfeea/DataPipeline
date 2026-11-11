"""Place the production target at the selected CuRobo source state after reset."""

from __future__ import annotations

import numpy as np
import torch
from configs.assets import SCENE_ENTITY_TARGET

from .replay import matrix_pose_wxyz
from .robot_adapter import pose_matrix


def set_source_target_pose(env, task: dict) -> list[float]:
    """Write the source state's E-frame pose to the live target and verify it."""

    target = env.scene[SCENE_ENTITY_TARGET]
    source = task["source"]
    world_from_environment = np.asarray(task["frames"]["T_WE"], dtype=np.float64)
    world_from_object = world_from_environment @ pose_matrix(
        source["position_e"], source["rotation_wxyz"]
    )
    pose_w = np.asarray(matrix_pose_wxyz(world_from_object), dtype=np.float64)
    device = target.data.root_pos_w.device
    target.write_root_pose_to_sim(
        torch.as_tensor(pose_w, dtype=target.data.root_pos_w.dtype, device=device).view(1, 7)
    )
    target.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(env.sim.get_physics_dt())
    measured = np.concatenate(
        (
            target.data.root_pos_w[0].detach().cpu().numpy(),
            target.data.root_quat_w[0].detach().cpu().numpy(),
        )
    ).astype(np.float64)
    if np.linalg.norm(measured[:3] - pose_w[:3]) > 1e-4:
        raise RuntimeError("production target did not reach selected source position")
    measured_q = measured[3:] / np.linalg.norm(measured[3:])
    if abs(float(np.dot(measured_q, pose_w[3:]))) < 1.0 - 1e-5:
        raise RuntimeError("production target did not reach selected source orientation")
    return measured.tolist()
