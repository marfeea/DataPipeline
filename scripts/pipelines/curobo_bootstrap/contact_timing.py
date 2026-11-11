"""Keep the 60 Hz control clock while varying contact physics substeps."""

from math import isfinite


def configure_contact_timing(cfg, *, physics_hz=120, control_hz=60.0, velocity_iterations=0):
    if type(physics_hz) is not int or physics_hz <= 0:
        raise ValueError("physics_hz must be a positive integer")
    if not isfinite(control_hz) or control_hz <= 0:
        raise ValueError("control_hz must be finite and positive")
    ratio = physics_hz / control_hz
    if ratio < 1 or not ratio.is_integer():
        raise ValueError("physics_hz must be an integer multiple of control_hz")
    if type(velocity_iterations) is not int or not 0 <= velocity_iterations <= 255:
        raise ValueError("velocity_iterations must be an integer in [0, 255]")
    if velocity_iterations and cfg.sim.physx.solver_type != 1:
        raise ValueError("accurate contact velocity updates require the TGS solver")
    cfg.sim.dt = 1.0 / physics_hz
    cfg.decimation = int(ratio)
    cfg.sim.render_interval = cfg.decimation
    contract = dict(physics_hz=physics_hz, control_hz=control_hz, physics_dt_s=cfg.sim.dt, decimation=cfg.decimation)
    if velocity_iterations:
        physx = cfg.sim.physx
        physx.min_velocity_iteration_count = max(physx.min_velocity_iteration_count, velocity_iterations)
        physx.max_velocity_iteration_count = max(physx.max_velocity_iteration_count, physx.min_velocity_iteration_count)
        physx.enable_external_forces_every_iteration = True
        contract["physx_solver"] = dict(
            solver_type=physx.solver_type,
            min_velocity_iteration_count=physx.min_velocity_iteration_count,
            max_velocity_iteration_count=physx.max_velocity_iteration_count,
            enable_external_forces_every_iteration=True,
        )
    return contract


def configure_flange_velocity_limit(cfg, *, robot_name, limit_rad_s=None):
    """Lower the physical wrist drive limit without changing trajectory acceptance."""
    if limit_rad_s is None:
        return None
    if not isfinite(limit_rad_s) or limit_rad_s <= 0:
        raise ValueError("contact flange velocity limit must be finite and positive")
    actuator = getattr(cfg.scene, robot_name).actuators["arm"]
    limits = actuator.velocity_limit_sim
    if not isinstance(limits, dict) or "Flange" not in limits:
        raise ValueError("contact flange limit requires named physical joint limits")
    original = limits["Flange"]
    if not isfinite(original) or limit_rad_s > original:
        raise ValueError("contact flange limit cannot exceed the original physical limit")
    actuator.velocity_limit_sim = dict(limits, Flange=limit_rad_s)
    return dict(joint="Flange", velocity_limit_rad_s=limit_rad_s, original_velocity_limit_rad_s=original)


def configure_wrist_velocity_limit(cfg, *, robot_name, limit_rad_s=None):
    """Use a conservative physical velocity cap for the three wrist joints."""
    if limit_rad_s is None:
        return None
    if not isfinite(limit_rad_s) or limit_rad_s <= 0:
        raise ValueError("contact wrist velocity limit must be finite and positive")
    actuator = getattr(cfg.scene, robot_name).actuators["arm"]
    limits = actuator.velocity_limit_sim
    names = ("Joint4", "Joint5", "Flange")
    if not isinstance(limits, dict) or not all(name in limits for name in names):
        raise ValueError("contact wrist limit requires named physical joint limits")
    original = {name: limits[name] for name in names}
    if any(not isfinite(value) or limit_rad_s > value for value in original.values()):
        raise ValueError("contact wrist limit cannot exceed any original physical limit")
    actuator.velocity_limit_sim = dict(limits, **dict.fromkeys(names, limit_rad_s))
    return dict(joint_names=list(names), velocity_limit_rad_s=limit_rad_s, original_velocity_limits_rad_s=original)
