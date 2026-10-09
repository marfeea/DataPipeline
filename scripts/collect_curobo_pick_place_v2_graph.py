"""Run and summarize all directed CuRobo V2 pick-place routes in Isaac Lab."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from DataPipe.pipelines.curobo_bootstrap.route_experiment import freeze_experiment, require_height, route_config
from DataPipe.pipelines.curobo_bootstrap.task_targets import LIFT_CLEARANCE_M
from DataPipe.state4_routing import add_state4_routing_args, routing_arguments, routing_from_args, routing_payload


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def has_environment_failure(run: dict) -> bool:
    markers = (
        "Ninja is required",
        "DLL load failed",
        "Failed to load the shared library",
        "WinError 32",
        "WinError 126",
    )
    latest = {stage["stage"]: stage for stage in run.get("stages", [])}
    for stage in latest.values():
        command = stage.get("command", [])
        if "--output" not in command:
            continue
        output = Path(command[command.index("--output") + 1])
        paths = [output] if output.suffix == ".json" else [output / "report.json", output / "plan_report.json"]
        reports = [read_json(path) for path in paths if path.exists()]
        if any(report.get("passed") for report in reports):
            continue
        log = Path(stage["log"])
        text = (
            "\n".join(str(report.get("failure_reason") or report.get("error") or "") for report in reports)
            if reports
            else log.read_text(encoding="utf-8", errors="replace")
            if log.exists()
            else ""
        )
        if any(marker in text for marker in markers):
            return True
    return False


def main() -> int:
    runtime = Path(sys.executable).parent
    os.environ["PATH"] = os.pathsep.join(
        [str(runtime), str(runtime / "Scripts"), str(runtime / "Library/bin"), os.environ.get("PATH", "")]
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--skip", nargs="*", default=[], help="Pairs already tested, e.g. 1:0")
    parser.add_argument("--seed-count", type=int, default=200)
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument("--contact-policy", choices=("strict", "run003"), default="strict")
    parser.add_argument("--lift-clearance-m", type=float, default=LIFT_CLEARANCE_M)
    parser.add_argument("--state4-lift-clearance-m", type=float)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--physics-hz", type=int, default=120)
    parser.add_argument("--capture-execution", action="store_true")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--collision-model", type=Path)
    parser.add_argument("--retry-environment-failures", action="store_true")
    add_state4_routing_args(parser)
    args = parser.parse_args()
    if args.physics_hz < 60 or args.physics_hz % 60:
        parser.error("physics-hz must be a positive integer multiple of 60")
    root = Path(__file__).resolve().parents[1]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    skipped = set(args.skip)
    configs = {
        f"{source}_to_{destination}": route_config(
            source,
            destination,
            lift_clearance_m=args.lift_clearance_m,
            state4_lift_clearance_m=args.state4_lift_clearance_m,
            state4_routing=routing_from_args(args),
        )
        for source in range(5)
        for destination in range(5)
        if source != destination
    }
    model = (args.collision_model or root / "configs/curobo/assets/aubo_e5_project/collision_model_v2.json").resolve()
    freeze_experiment(
        output_root,
        {
            "schema": "curobo_route_experiment/1",
            "route_lift_clearance_m": {name: config.lift_clearance_m for name, config in configs.items()},
            "route_state4_routing": {
                name: routing_payload(config.state4_routing)
                for name, config in configs.items()
            },
            "collision_model": str(model),
            "collision_model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            "contact_policy": args.contact_policy,
            "seed_count": args.seed_count,
            "max_candidates": args.max_candidates,
            "mode": "gui" if args.gui else "headless",
            "camera_resolution": [args.camera_width, args.camera_height],
            "physics_hz": args.physics_hz,
            "capture_execution": args.capture_execution,
        },
    )
    if args.gui:
        os.environ["HEADLESS"] = "0"
    summary_path = output_root / "graph_summary.json"
    summary = read_json(summary_path) or {"schema": "curobo_v2_directed_graph/1", "routes": {}}
    if summary.get("contact_policy", args.contact_policy) != args.contact_policy:
        raise ValueError("cannot resume graph with a different contact policy")
    summary["contact_policy"] = args.contact_policy

    for source in range(5):
        for destination in range(5):
            if source == destination:
                continue
            key = f"{source}:{destination}"
            name = f"{source}_to_{destination}"
            if key in skipped:
                print(f"SKIP {name}: separately tested", flush=True)
                continue
            output = output_root / name
            evidence = output.with_name(output.name + "_evidence")
            run_report = evidence / "run_report.json"
            if args.retry_environment_failures:
                existing = read_json(run_report)
                environment_failure = bool(
                    existing and not existing.get("passed") and has_environment_failure(existing)
                )
                if environment_failure:
                    output = output_root / (name + "_retry")
                    evidence = output.with_name(output.name + "_evidence")
                    run_report = evidence / "run_report.json"
                    print(f"RETRY {name}: runtime environment failure", flush=True)
            if not output.exists() and not evidence.exists():
                command = [
                    sys.executable,
                    str(root / "scripts" / "collect_isaacsim.py"),
                    "--source-index",
                    str(source),
                    "--destination-index",
                    str(destination),
                    "--device",
                    "cuda:0",
                    "--output",
                    str(output),
                    "--seed-count",
                    str(args.seed_count),
                    "--max-candidates",
                    str(args.max_candidates),
                    "--contact-policy",
                    args.contact_policy,
                    "--lift-clearance-m",
                    str(configs[name].lift_clearance_m),
                    "--physics-hz",
                    str(args.physics_hz),
                    *routing_arguments(configs[name].state4_routing),
                ]
                if args.capture_execution:
                    command.append("--capture-execution")
                if args.collision_model is not None:
                    command.extend(["--collision-model", str(args.collision_model.resolve())])
                if args.gui:
                    command.append("--gui")
                print(f"START {name}", flush=True)
                with (output_root / f"{name}.log").open("w", encoding="utf-8") as stream:
                    result = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=False)
                print(f"PLANNED {name} exit={result.returncode}", flush=True)
            run = read_json(run_report)
            if run is None:
                route = {
                    "source_index": source,
                    "destination_index": destination,
                    "passed": False,
                    "stage": "missing_run_report",
                    "failure_reason": "no run report produced",
                }
            else:
                targets = read_json(evidence / "targets.json")
                if targets and "task_snapshot" in targets:
                    require_height(targets["task_snapshot"], configs[name].lift_clearance_m)
                    expected_routing = routing_payload(configs[name].state4_routing)
                    if targets["task_snapshot"].get("state4_routing") != expected_routing:
                        raise ValueError("restored state4 routing differs from frozen experiment")
                route = {
                    "source_index": source,
                    "destination_index": destination,
                    "source_name": run.get("source_name"),
                    "destination_name": run.get("destination_name"),
                    "task_hash": run.get("task_hash"),
                    "world_hash": run.get("world_hash"),
                    "lift_clearance_m": configs[name].lift_clearance_m,
                    "passed": bool(run.get("passed")),
                    "stage": run.get("stage"),
                    "failure_reason": run.get("failure_reason"),
                    "run_report": str(run_report),
                    "trajectory": str(output / "trajectory.npz") if run.get("passed") else None,
                }
            if route["passed"]:
                replay_path = output / "replay_report.json"
                if not replay_path.exists():
                    command = [
                        sys.executable,
                        str(root / "scripts" / "replay_isaacsim.py"),
                        "--run",
                        str(output),
                        "--device",
                        "cuda:0",
                        "--camera-width",
                        str(args.camera_width),
                        "--camera-height",
                        str(args.camera_height),
                    ]
                    if not args.gui:
                        command.extend(["--headless", "--no-realtime"])
                    print(f"REPLAY {name}", flush=True)
                    with (output_root / f"{name}_replay.log").open("w", encoding="utf-8") as stream:
                        result = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=False)
                    print(f"REPLAYED {name} exit={result.returncode}", flush=True)
                replay = read_json(replay_path)
                route["replay_passed"] = bool(replay and replay.get("passed"))
                route["replay_report"] = str(replay_path) if replay else None
                route["visualization_dir"] = replay.get("visualization_dir") if replay else None
            summary["routes"][name] = route
            summary_path.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            print(
                f"RESULT {name} plan={route['passed']} replay={route.get('replay_passed')} stage={route['stage']}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
