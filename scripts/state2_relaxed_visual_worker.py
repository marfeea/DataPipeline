"""Process-local diagnostic overrides for state2; production files stay unchanged."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

POLICY = {
    "name": "state2_relaxed_visual/1",
    "diagnostic_only": True,
    "source_index": 2,
    "self_collision_hard_gate": False,
    "velocity_acceptance_hard_gate": False,
    "world_collision_hard_gate": True,
    "joint_position_hard_gate": True,
    "physical_drive_limits_changed": False,
    "legacy_acceleration_limit_rad_s2": 50.0,
    "capture_interval_control_frames": 1,
}


def relaxed_acceptance(strict_result):
    """Preserve every raw check and metric; ignore only two rejection decisions."""
    ignored = ("self_collision_free", "velocity_limits")
    result = dict(strict_result)
    result["strict_passed"] = bool(strict_result["passed"])
    result["diagnostic_ignored_checks"] = list(ignored)
    result["passed"] = all(value for key, value in strict_result["checks"].items() if key not in ignored)
    result["diagnostic_policy"] = POLICY
    return result


def install_overrides():
    from DataPipe.pipelines.curobo_bootstrap import planner as planning
    from DataPipe.pipelines.curobo_bootstrap import replay_visualization as visualization

    original_planner = planning.CuroboMvpPlanner
    original_validate = planning.validate_trajectory

    class State2DiagnosticPlanner(original_planner):
        def __init__(self, *args, **kwargs):
            from curobo.wrap.reacher.motion_gen import MotionGenConfig

            descriptor = vars(MotionGenConfig)["load_from_robot_config"]
            original_loader = MotionGenConfig.load_from_robot_config

            def diagnostic_loader(*loader_args, **loader_kwargs):
                loader_kwargs["self_collision_check"] = False
                loader_kwargs["self_collision_opt"] = False
                return original_loader(*loader_args, **loader_kwargs)

            # Only this worker's constructor sees the altered MotionGen flags.
            MotionGenConfig.load_from_robot_config = staticmethod(diagnostic_loader)
            try:
                super().__init__(*args, **kwargs)
            finally:
                MotionGenConfig.load_from_robot_config = descriptor

        def endpoint_checks(self, start_q, goal_q):
            checks = super().endpoint_checks(start_q, goal_q)
            checks["strict_self_collision_free"] = checks["self_collision_free"]
            checks["self_collision_free"] = True  # effective diagnostic decision, not a measured score
            checks["diagnostic_policy"] = POLICY
            return checks

    def diagnostic_validate(*args, **kwargs):
        return relaxed_acceptance(original_validate(*args, **kwargs))

    planning.CuroboMvpPlanner = State2DiagnosticPlanner
    planning.validate_trajectory = diagnostic_validate
    visualization.FRAME_INTERVAL = 1


def stamp(value):
    if isinstance(value, dict):
        if "self_collision_hard_gate" in value:
            value["self_collision_hard_gate"] = False
        for child in value.values():
            stamp(child)
    elif isinstance(value, list):
        for child in value:
            stamp(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("branches", "noncontact", "contact"), required=True)
    parser.add_argument("stage_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.stage_arguments
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    targets_path = Path(arguments[arguments.index("--targets-report") + 1])
    targets = json.loads(targets_path.read_text(encoding="utf-8"))
    if targets["source_index"] != 2 or targets["task_snapshot"]["source_index"] != 2:
        raise ValueError("Diagnostic overrides are restricted to source state2")
    output = Path(arguments[arguments.index("--output") + 1]).resolve()
    scripts = {
        "branches": "audit_curobo_v2_pre_pick_branches.py",
        "noncontact": "probe_curobo_v2_noncontact.py",
        "contact": "probe_curobo_v2_legacy_contact_isaac.py",
    }
    install_overrides()
    script = PROJECT / "scripts" / scripts[args.kind]
    sys.argv = [str(script), *arguments]
    source = script.read_text(encoding="utf-8")
    if args.kind in ("noncontact", "contact"):
        marker = "report = {"
        if source.count(marker) != 1:
            raise ValueError("Production report initializer changed; review migration before running")
        source = source.replace(marker, "report = {\n        'diagnostic_policy': STATE2_DIAGNOSTIC_POLICY,\n        'formal_accepted': False,", 1)
        source = source.replace('"self_collision_hard_gate": True', '"self_collision_hard_gate": False')
    if args.kind == "contact":
        source = source.replace("camera_manifest.update(\n", "camera_manifest.update(\n                diagnostic_policy=STATE2_DIAGNOSTIC_POLICY,\n                formal_accepted=False,\n", 1)
    migration_dir = output.parent / "migrated_sources"
    migration_dir.mkdir(exist_ok=True)
    (migration_dir / script.name).write_text(source, encoding="utf-8")
    code = 0
    try:
        # __file__ keeps the production project's asset/import path resolution;
        # the migrated code is evaluated only inside this diagnostic worker.
        exec(compile(source, str(script), "exec"), {
            "__name__": "__main__", "__file__": str(script),
            "STATE2_DIAGNOSTIC_POLICY": POLICY,
        })
    except SystemExit as error:
        code = int(error.code or 0)
    paths = [output] if output.suffix == ".json" else [output / "report.json"]
    if args.kind == "contact":
        paths += [output / "execution_visualization/frames.json"]
    for path in paths:
        if not path.exists():
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if args.kind == "branches":
            value["strict_passed"] = value["passed"]
            for candidate in value["candidates"]:
                candidate["strict_collision_free"] = candidate["collision_free"]
                candidate["collision_free"] = candidate["world_collision_score"] <= 0
            value["candidates"].sort(key=lambda c: (not c["collision_free"], c["l2_displacement_from_start_rad"]))
            value["collision_free_count"] = sum(c["collision_free"] for c in value["candidates"])
            value["passed"] = value["start_world_collision_score"] <= 0 and value["collision_free_count"] > 0
        stamp(value)
        value["diagnostic_policy"] = POLICY
        value["formal_accepted"] = False
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
