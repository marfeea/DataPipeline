from scripts.plan_curobo_pick_place_v2 import _stage_command


def test_gui_removes_headless_from_isaac_stage_and_preserves_task_arguments():
    command = _stage_command(
        "python.exe",
        "targets.py",
        ("--source-index", 3, "--destination-index", 0, "--output", "targets.json", "--headless"),
        gui=True,
    )
    assert command == [
        "python.exe",
        "targets.py",
        "--source-index",
        "3",
        "--destination-index",
        "0",
        "--output",
        "targets.json",
    ]


def test_default_execution_keeps_headless_flag():
    assert _stage_command("python.exe", "contact.py", ("--headless",), gui=False) == [
        "python.exe",
        "contact.py",
        "--headless",
    ]


def test_gui_does_not_change_non_isaac_planner_arguments():
    arguments = ("--disable-cuda-graph", "--graph-first", "--max-attempts", 4)
    assert _stage_command("python.exe", "probe.py", arguments, gui=True) == _stage_command(
        "python.exe",
        "probe.py",
        arguments,
        gui=False,
    )
