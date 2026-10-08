"""Lazy, version-checked CuRobo GPU FK for P3; no fallback or collision claims."""

from importlib.metadata import version
from pathlib import Path

from .robot_adapter import pose_matrix


class CuroboFk:
    def __init__(self, urdf, *, joint_names, ee_link="project_tcp", device="cuda:0"):
        self.version = version("nvidia_curobo")
        if self.version != "0.7.8":
            raise ValueError(f"CuRobo adapter requires tested candidate 0.7.8, got {self.version}")
        import torch

        from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
        from curobo.types.base import TensorDeviceType

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("CuRobo FK requires a working CUDA device")
        cfg = CudaRobotModelConfig.from_basic_urdf(
            str(Path(urdf).resolve()), "AUBO_E5", ee_link, TensorDeviceType(device=self.device)
        )
        self.model = CudaRobotModel(cfg)
        if tuple(self.model.joint_names) != tuple(joint_names):
            raise ValueError(f"CuRobo joint names/order mismatch: {self.model.joint_names}")
        limits = cfg.get_joint_limits()
        if limits.position.shape != (2, 6) or limits.velocity.shape != (2, 6):
            raise ValueError("unexpected CuRobo 0.7.8 joint limit layout")
        self.contract = {
            "joint_names": self.model.joint_names,
            "lower": limits.position[0].cpu().tolist(),
            "upper": limits.position[1].cpu().tolist(),
            "velocity": limits.velocity[1].cpu().tolist(),
        }

    def __call__(self, q):
        with self.torch.inference_mode():
            state = self.model.get_state(
                self.torch.as_tensor(q, dtype=self.torch.float32, device=self.device).view(1, 6)
            )
            return pose_matrix(state.ee_position[0].cpu().numpy(), state.ee_quaternion[0].cpu().numpy())
