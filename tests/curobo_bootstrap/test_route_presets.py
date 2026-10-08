import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from configs.curobo.route_presets import configure_route_preset, load_route_preset, validate_preset_route
from DataPipe.pipelines.curobo_bootstrap.task_targets import (
    PickPlaceV2Config,
    _rotation_wxyz_from_matrix,
    task_snapshot,
)
from DataPipe.source2_routing import add_source2_routing_args, source2_routing_from_args
from DataPipe.state4_routing import add_state4_routing_args, routing_from_args


def parse(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-index", type=int, required=True)
    parser.add_argument("--destination-index", type=int, required=True)
    parser.add_argument("--lift-clearance-m", type=float, default=0.4)
    parser.add_argument("--collision-model", type=Path)
    add_state4_routing_args(parser)
    add_source2_routing_args(parser)
    preset = configure_route_preset(parser, argv)
    args = parser.parse_args(argv)
    validate_preset_route(parser, args, preset)
    return args


@pytest.mark.parametrize("destination", (0, 1, 2, 3))
def test_successful_preset_reproduces_frozen_state4_route_geometry(destination):
    route = f"4_to_{destination}"
    args = parse(["--route-preset", route])
    assert (args.source_index, args.destination_index) == (4, destination)
    assert args.gui and args.capture_execution and args.physics_hz == 120
    assert args.max_candidates == 8 and args.seed_count == 200 and args.contact_policy == "run003"
    project = Path(__file__).resolve().parents[2]
    preset = load_route_preset(route)
    accepted = json.loads((project / preset["validation"]["accepted_run"] / "scene_snapshot.json").read_text())
    task = accepted["task"]
    config = PickPlaceV2Config(
        source_index=4,
        destination_index=destination,
        lift_clearance_m=args.lift_clearance_m,
        state4_routing=routing_from_args(args),
    )
    transform = np.asarray(task["frames"]["T_WB"])
    snapshot = task_snapshot(
        config=config,
        environment_origin_w=np.asarray(task["frames"]["T_WE"])[:3, 3],
        base_pose_w=(*transform[:3, 3], *_rotation_wxyz_from_matrix(transform)),
    )
    for field in ("waypoint_position_b", "contact_waypoint_position_b"):
        assert set(snapshot[field]) == set(task[field])
        assert all(np.allclose(value, task[field][name], atol=1e-12, rtol=0) for name, value in snapshot[field].items())


def test_preset_does_not_change_regular_routes_and_allows_explicit_experiment_override():
    normal = parse(["--source-index", "3", "--destination-index", "1"])
    assert normal.lift_clearance_m == 0.4 and not normal.state4_routing
    override = parse(["--route-preset", "4_to_0", "--state4-pick-lift-m", "0.02"])
    assert override.state4_pick_lift_m == 0.02 and override.lift_clearance_m == 0.3


def test_success_preset_cannot_be_applied_to_unvalidated_destination():
    with pytest.raises(SystemExit):
        parse(["--route-preset", "4_to_0", "--destination-index", "4"])
    with pytest.raises(ValueError, match="unknown route preset"):
        load_route_preset("unvalidated_route")


@pytest.mark.parametrize("source", (0, 1, 3))
def test_incoming_success_presets_reproduce_their_accepted_task_geometry(source):
    route = f"{source}_to_4"
    args = parse(["--route-preset", route])
    preset = load_route_preset(route)
    project = Path(__file__).resolve().parents[2]
    task = json.loads((project / preset["validation"]["accepted_run"] / "scene_snapshot.json").read_text())["task"]
    assert preset["validation"]["task_hash"] == task["task_hash"]
    transform = np.asarray(task["frames"]["T_WB"])
    current = task_snapshot(
        config=PickPlaceV2Config(
            source_index=source, destination_index=4, lift_clearance_m=args.lift_clearance_m,
            state4_routing=routing_from_args(args), source2_routing=source2_routing_from_args(args),
        ),
        environment_origin_w=np.asarray(task["frames"]["T_WE"])[:3, 3],
        base_pose_w=(*transform[:3, 3], *_rotation_wxyz_from_matrix(transform)),
    )
    for field in ("waypoint_position_b", "contact_waypoint_position_b"):
        assert current[field].keys() == task[field].keys()
        assert all(np.allclose(point, task[field][name], atol=1e-12, rtol=0) for name, point in current[field].items())
