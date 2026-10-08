"""Load validated route defaults while preserving explicit CLI parameter overrides."""

import argparse
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRESET_FILE = Path(__file__).with_suffix(".json")
PARAMETER_KEYS = {
    "source_index",
    "destination_index",
    "lift_clearance_m",
    "state4_routing",
    "state4_pick_lift_m",
    "state4_pick_exit_m",
    "state4_place_standoff_m",
    "physics_hz",
    "gui",
    "capture_execution",
    "seed_count",
    "max_candidates",
    "contact_policy",
    "collision_model",
}
OPTIONAL_PARAMETER_KEYS = {
    "state4_place_entry_lift_m", "state4_place_measured_grasp",
    "state2_pre_pick_side_m", "state2_pre_pick_lift_m",
    "contact_velocity_iterations",
    "contact_flange_velocity_limit_rad_s",
    "contact_wrist_velocity_limit_rad_s",
}


def load_route_preset(name, *, path=PRESET_FILE):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != "curobo_route_presets/1":
        raise ValueError("unsupported route preset schema")
    if name not in data["routes"]:
        raise ValueError(f"unknown route preset: {name}")
    preset = data["routes"][name]
    parameters = dict(preset["parameters"])
    if not set(parameters) >= PARAMETER_KEYS or set(parameters) - PARAMETER_KEYS - OPTIONAL_PARAMETER_KEYS:
        raise ValueError("route preset parameter keys differ from supported runtime settings")
    if name != f"{parameters['source_index']}_to_{parameters['destination_index']}":
        raise ValueError("route preset name differs from its state indices")
    parameters["collision_model"] = (PROJECT_ROOT / parameters["collision_model"]).resolve()
    if hashlib.sha256(parameters["collision_model"].read_bytes()).hexdigest() != preset["collision_model_sha256"]:
        raise ValueError("route preset collision model differs from validated evidence")
    return dict(preset, name=name, config_path=str(Path(path).resolve()), parameters=parameters)


def configure_route_preset(parser, argv=None):
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--route-preset")
    selected, _ = selector.parse_known_args(argv)
    parser.add_argument("--route-preset", help="Load a validated route from configs/curobo/route_presets.json")
    if selected.route_preset is None:
        return None
    try:
        preset = load_route_preset(selected.route_preset)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    parser.set_defaults(**preset["parameters"])
    for action in parser._actions:
        if action.dest in ("source_index", "destination_index"):
            action.required = False
    return preset


def validate_preset_route(parser, args, preset):
    if preset is not None and (args.source_index, args.destination_index) != (
        preset["parameters"]["source_index"],
        preset["parameters"]["destination_index"],
    ):
        parser.error("route preset is validated only for its configured source and destination")
