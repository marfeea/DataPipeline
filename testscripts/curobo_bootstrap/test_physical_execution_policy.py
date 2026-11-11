"""Offline control-flow checks; these tests never launch a simulator."""

import pytest

from scripts.pipelines.curobo_bootstrap.physical_execution import execute_physical_route


@pytest.mark.parametrize("continue_on_failure", [False, True])
@pytest.mark.parametrize("terminal_index", [1, 3])
def test_task_terminal_does_not_truncate_requested_full_route(continue_on_failure, terminal_index):
    segments = ["reach", "legacy_lift", "legacy_lift", "free_space_transfer", "legacy_place_approach",
                "open_gripper", "withdraw"]
    calls = []
    done = False

    def execute(index, **kwargs):
        nonlocal done
        calls.append((index, kwargs.get("stage", segments[index])))
        done = done or index == terminal_index

    last, placed = execute_physical_route(
        segments, execute, lift_hold_s=0.35, task_done=lambda: done, should_abort=lambda: False,
        continue_on_failure=continue_on_failure,
    )
    assert not placed
    if continue_on_failure:
        assert last == len(segments) - 1
        assert [stage for _, stage in calls][-3:] == ["legacy_place_approach", "open_gripper", "withdraw"]
        assert sum(stage == "physical_lift_hold" for _, stage in calls) == 21
    else:
        assert last == terminal_index
        assert not any(stage in {"legacy_place_approach", "open_gripper", "withdraw"} for _, stage in calls)


def test_terminated_task_still_reaches_replacement_placement_controller():
    calls = []
    last, placed = execute_physical_route(
        ["reach", "legacy_lift", "free_space_transfer", "legacy_place_approach", "open_gripper"],
        lambda index, **kwargs: calls.append((index, kwargs)), lift_hold_s=0.35,
        task_done=lambda: True, should_abort=lambda: False, continue_on_failure=True,
        place=lambda index: calls.append((index, "replacement_placement")),
    )
    assert placed and last == 3
    assert calls[-1] == (3, "replacement_placement")


def test_window_closed_still_stops_physical_execution():
    with pytest.raises(RuntimeError, match="SimulationApp stopped"):
        execute_physical_route(["reach", "legacy_lift"], lambda *args, **kwargs: pytest.fail("no step"),
                               lift_hold_s=0.35, task_done=lambda: False, should_abort=lambda: True,
                               continue_on_failure=True)
