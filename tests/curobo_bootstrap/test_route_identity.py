"""Route identity gates for selected CuRobo pick-and-place targets."""

import pytest

from DataPipe.pipelines.curobo_bootstrap.route_identity import (
    require_report_identity,
    task_identity,
)
from DataPipe.pipelines.curobo_bootstrap.task_targets import PickPlaceV2Config, task_snapshot


def _task(source, destination):
    return task_snapshot(
        config=PickPlaceV2Config(source_index=source, destination_index=destination),
        environment_origin_w=(0.0, 0.0, 0.0),
        base_pose_w=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    )


def test_nondefault_route_identity_and_cross_route_rejection():
    selected = _task(1, 0)
    wrong = _task(0, 1)
    identity = task_identity(selected)
    assert identity["source_name"] == "sample_bottle_state_02"
    assert identity["destination_name"] == "sample_bottle_state_01"
    require_report_identity({"passed": True, **identity}, selected, label="probe")
    with pytest.raises(ValueError, match="differs from selected task"):
        require_report_identity({"passed": True, **task_identity(wrong)}, selected, label="probe")


@pytest.mark.parametrize("source,destination", [(-1, 0), (5, 0), (0, 5), (2, 2)])
def test_invalid_selected_route_rejected(source, destination):
    with pytest.raises(ValueError):
        PickPlaceV2Config(source_index=source, destination_index=destination)


def test_tampered_target_or_hash_rejected():
    task = _task(3, 2)
    task["source"]["name"] = "sample_bottle_state_01"
    with pytest.raises(ValueError, match="source differs"):
        task_identity(task)
    task = _task(3, 2)
    task["task_hash"] = _task(0, 1)["task_hash"]
    with pytest.raises(ValueError, match="hash mismatch"):
        task_identity(task)
