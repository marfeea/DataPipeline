from types import SimpleNamespace

import pytest

from DataPipe.pipelines.curobo_bootstrap.contact_timing import (
    configure_contact_timing,
    configure_flange_velocity_limit,
    configure_wrist_velocity_limit,
)


@pytest.mark.parametrize("physics_hz,substeps", [(120, 2), (240, 4), (480, 8)])
def test_more_physics_substeps_preserve_control_and_render_frequency(physics_hz, substeps):
    cfg = SimpleNamespace(sim=SimpleNamespace(dt=0, render_interval=0), decimation=0)
    contract = configure_contact_timing(cfg, physics_hz=physics_hz)
    assert cfg.decimation == cfg.sim.render_interval == substeps
    assert cfg.sim.dt * cfg.decimation == pytest.approx(1 / 60)
    assert contract["physics_hz"] == physics_hz and contract["control_hz"] == 60


@pytest.mark.parametrize("physics_hz", [0, -120, 30, 100, True, 240.0])
def test_invalid_or_nonintegral_control_clock_is_rejected(physics_hz):
    cfg = SimpleNamespace(sim=SimpleNamespace(dt=1 / 120, render_interval=2), decimation=2)
    with pytest.raises(ValueError):
        configure_contact_timing(cfg, physics_hz=physics_hz)
    assert cfg.sim.dt == 1 / 120 and cfg.decimation == 2


def test_accurate_velocity_solver_preserves_clock_and_existing_stricter_iterations():
    physx = SimpleNamespace(
        solver_type=1, min_velocity_iteration_count=4, max_velocity_iteration_count=255,
        enable_external_forces_every_iteration=False,
    )
    cfg = SimpleNamespace(sim=SimpleNamespace(dt=0, render_interval=0, physx=physx), decimation=0)
    contract = configure_contact_timing(cfg, velocity_iterations=2)
    assert cfg.sim.dt == 1 / 120 and cfg.decimation == 2
    assert physx.min_velocity_iteration_count == 4 and physx.max_velocity_iteration_count == 255
    assert physx.enable_external_forces_every_iteration
    assert contract["physx_solver"]["min_velocity_iteration_count"] == 4


@pytest.mark.parametrize("iterations", [True, 2.0, -1, 256])
def test_invalid_velocity_solver_settings_are_rejected_before_clock_changes(iterations):
    cfg = SimpleNamespace(sim=SimpleNamespace(dt=1 / 120, render_interval=2), decimation=2)
    with pytest.raises(ValueError):
        configure_contact_timing(cfg, physics_hz=240, velocity_iterations=iterations)
    assert cfg.sim.dt == 1 / 120 and cfg.decimation == 2


def test_velocity_solver_does_not_silently_enable_a_tgs_flag_for_pgs():
    cfg = SimpleNamespace(
        sim=SimpleNamespace(dt=1 / 120, render_interval=2, physx=SimpleNamespace(solver_type=0)), decimation=2
    )
    with pytest.raises(ValueError, match="TGS"):
        configure_contact_timing(cfg, velocity_iterations=2)


def wrist_config():
    limits = dict(Joint1=2.617994, Joint2=2.617994, Joint3=2.617994, Joint4=3.14, Joint5=3.14, Flange=3.14)
    actuator = SimpleNamespace(velocity_limit_sim=limits)
    return SimpleNamespace(scene=SimpleNamespace(robot=SimpleNamespace(actuators={"arm": actuator}))), limits


def test_lower_flange_limit_copies_limits_and_leaves_other_joints_unchanged():
    cfg, original = wrist_config()
    assert configure_flange_velocity_limit(cfg, robot_name="robot") is None
    contract = configure_flange_velocity_limit(cfg, robot_name="robot", limit_rad_s=0.4)
    limits = cfg.scene.robot.actuators["arm"].velocity_limit_sim
    assert limits["Flange"] == 0.4 and original["Flange"] == 3.14
    assert all(limits[name] == value for name, value in original.items() if name != "Flange")
    assert contract["original_velocity_limit_rad_s"] == 3.14


@pytest.mark.parametrize("limit", [0, -0.4, float("nan"), float("inf"), 3.15])
def test_flange_limit_rejects_invalid_or_higher_limits_without_mutation(limit):
    cfg, original = wrist_config()
    with pytest.raises(ValueError):
        configure_flange_velocity_limit(cfg, robot_name="robot", limit_rad_s=limit)
    assert cfg.scene.robot.actuators["arm"].velocity_limit_sim is original


def test_wrist_cap_is_applied_to_all_three_wrist_joints_without_replacing_arm_limits():
    cfg, original = wrist_config()
    report = configure_wrist_velocity_limit(cfg, robot_name="robot", limit_rad_s=0.7)
    limits = cfg.scene.robot.actuators["arm"].velocity_limit_sim
    assert all(limits[name] == 0.7 for name in ("Joint4", "Joint5", "Flange"))
    assert all(limits[name] == original[name] for name in ("Joint1", "Joint2", "Joint3"))
    assert all(value == 3.14 for value in report["original_velocity_limits_rad_s"].values())
    cfg, original = wrist_config()
    with pytest.raises(ValueError):
        configure_wrist_velocity_limit(cfg, robot_name="robot", limit_rad_s=3.15)
    assert cfg.scene.robot.actuators["arm"].velocity_limit_sim is original
