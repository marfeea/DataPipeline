"""Generate route-specific CuRobo V2 evidence, hybrid trajectory, and failure report."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from math import isfinite
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.curobo.route_presets import configure_route_preset, validate_preset_route
from configs.task import TARGET_STATES
from DataPipe.pipelines.curobo_bootstrap.pick_place_artifact import verify_hybrid_artifact
from DataPipe.pipelines.curobo_bootstrap.route_experiment import require_height
from DataPipe.pipelines.curobo_bootstrap.route_identity import (
    require_report_identity,
    task_identity,
)
from DataPipe.pipelines.curobo_bootstrap.task_targets import LIFT_CLEARANCE_M, PickPlaceV2Config
from DataPipe.source2_routing import add_source2_routing_args, source2_routing_arguments, source2_routing_from_args
from DataPipe.state4_routing import add_state4_routing_args, routing_arguments, routing_from_args, routing_payload


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _stage_command(executable, script, arguments, *, gui: bool):
    """Preserve stage arguments while opting Isaac stages into visible execution."""
    return [
        str(executable),
        str(script),
        *(str(item) for item in arguments if not (gui and str(item) == "--headless")),
    ]


def main() -> int:
    runtime = Path(sys.executable).parent
    os.environ["PATH"] = os.pathsep.join(
        [str(runtime), str(runtime / "Scripts"), str(runtime / "Library/bin"), os.environ.get("PATH", "")]
    )
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="TARGET_STATES: " + ", ".join(f"{index}={state.name}" for index, state in enumerate(TARGET_STATES)),
    )
    parser.add_argument("--source-index", type=int, required=True)
    parser.add_argument("--destination-index", type=int, required=True)
    parser.add_argument("--lift-clearance-m", type=float, default=LIFT_CLEARANCE_M)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gui", action="store_true", help="Run Isaac target/world/contact stages with visible windows")
    parser.add_argument("--physics-hz", type=int, default=120)
    parser.add_argument("--contact-velocity-iterations", type=int, default=0)
    parser.add_argument("--contact-flange-velocity-limit-rad-s", type=float)
    parser.add_argument("--contact-wrist-velocity-limit-rad-s", type=float)
    parser.add_argument("--capture-execution", action="store_true", help="Save live RGB for every contact attempt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-count", type=int, default=200)
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument("--contact-policy", choices=("strict", "run003"), default="strict")
    parser.add_argument(
        "--collision-model", type=Path, default=root / "configs/curobo/assets/aubo_e5_project/collision_model_v2.json"
    )
    add_state4_routing_args(parser)
    add_source2_routing_args(parser)
    preset = configure_route_preset(parser)
    args = parser.parse_args()
    validate_preset_route(parser, args, preset)
    if args.gui:
        os.environ["HEADLESS"] = "0"
    try:
        config = PickPlaceV2Config(
            source_index=args.source_index,
            destination_index=args.destination_index,
            lift_clearance_m=args.lift_clearance_m,
            state4_routing=routing_from_args(args),
            source2_routing=source2_routing_from_args(args),
        )
    except ValueError as error:
        parser.error(str(error))
    if args.device != "cuda:0":
        parser.error("CuRobo V2 was validated on cuda:0")
    if args.seed_count < 1 or args.max_candidates < 1:
        parser.error("seed-count and max-candidates must be positive")
    if args.physics_hz < 60 or args.physics_hz % 60:
        parser.error("physics-hz must be a positive integer multiple of 60")
    if not 0 <= args.contact_velocity_iterations <= 255:
        parser.error("contact-velocity-iterations must be in [0, 255]")
    if args.contact_flange_velocity_limit_rad_s is not None and (
        not isfinite(args.contact_flange_velocity_limit_rad_s)
        or not 0 < args.contact_flange_velocity_limit_rad_s <= 3.14
    ):
        parser.error("contact-flange-velocity-limit-rad-s must be finite and in (0, 3.14]")
    if args.contact_flange_velocity_limit_rad_s is not None and args.contact_wrist_velocity_limit_rad_s is not None:
        parser.error("select either the flange limit or the wrist limit")
    if args.contact_wrist_velocity_limit_rad_s is not None and (
        not isfinite(args.contact_wrist_velocity_limit_rad_s) or not 0 < args.contact_wrist_velocity_limit_rad_s <= 3.14
    ):
        parser.error("contact-wrist-velocity-limit-rad-s must be finite and in (0, 3.14]")
    output = args.output.resolve()
    evidence = output.with_name(output.name + "_evidence")
    if output.exists() or evidence.exists():
        raise FileExistsError("selected CuRobo output or its evidence sibling already exists")
    evidence.mkdir(parents=True)
    status_path = evidence / "run_report.json"
    report = {
        "schema": "curobo_pick_place_v2_route_run/1",
        "passed": False,
        "stage": "startup",
        "source_index": config.source_index,
        "destination_index": config.destination_index,
        "source_name": TARGET_STATES[config.source_index].name,
        "destination_name": TARGET_STATES[config.destination_index].name,
        "output": str(output),
        "evidence": str(evidence),
        "device": args.device,
        "mode": "gui" if args.gui else "headless",
        "lift_clearance_m": config.lift_clearance_m,
        "physics_hz": args.physics_hz,
        "capture_execution": args.capture_execution,
        "state4_routing": routing_payload(config.state4_routing),
        "stages": [],
        "contact_policy": args.contact_policy,
        "collision_model": str(args.collision_model.resolve()),
    }
    if preset is not None:
        report["route_preset"] = preset["name"]
        report["route_preset_config"] = preset["config_path"]
    if config.source2_routing is not None:
        report["source2_routing"] = vars(config.source2_routing)
    if args.contact_velocity_iterations:
        report["contact_velocity_iterations"] = args.contact_velocity_iterations
    if args.contact_flange_velocity_limit_rad_s is not None:
        report["contact_flange_velocity_limit_rad_s"] = args.contact_flange_velocity_limit_rad_s
    if args.contact_wrist_velocity_limit_rad_s is not None:
        report["contact_wrist_velocity_limit_rad_s"] = args.contact_wrist_velocity_limit_rad_s

    def save_status() -> None:
        status_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def invoke(stage: str, script: str, *arguments: object) -> int:
        report["stage"] = stage
        save_status()
        command = _stage_command(sys.executable, root / "scripts" / script, arguments, gui=args.gui)
        if script == "probe_curobo_v2_legacy_contact_isaac.py":
            command.extend(["--physics-hz", str(args.physics_hz)])
            if args.contact_velocity_iterations:
                command.extend(["--contact-velocity-iterations", str(args.contact_velocity_iterations)])
            if args.contact_flange_velocity_limit_rad_s is not None:
                command.extend(["--contact-flange-velocity-limit-rad-s", str(args.contact_flange_velocity_limit_rad_s)])
            if args.contact_wrist_velocity_limit_rad_s is not None:
                command.extend(["--contact-wrist-velocity-limit-rad-s", str(args.contact_wrist_velocity_limit_rad_s)])
            if args.capture_execution:
                command.append("--capture-execution")
        runtime_markers = (
            "Ninja is required",
            "DLL load failed",
            "Failed to load the shared library",
            "WinError 32",
            "WinError 126",
        )
        for attempt in range(1, 4):
            log = evidence / (f"{stage}.log" if attempt == 1 else f"{stage}_runtime_retry_{attempt}.log")
            print(
                f"CUROBO_ROUTE_STAGE route={config.source_index}_to_{config.destination_index} "
                f"stage={stage} runtime_attempt={attempt}",
                flush=True,
            )
            with log.open("w", encoding="utf-8", errors="replace") as stream:
                result = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=False)
            record = {
                "stage": stage,
                "runtime_attempt": attempt,
                "returncode": result.returncode,
                "log": str(log),
                "command": command,
            }
            report["stages"].append(record)
            save_status()
            output_path = Path(command[command.index("--output") + 1]).resolve()
            result_paths = (
                [output_path]
                if output_path.suffix == ".json"
                else [output_path / "report.json", output_path / "plan_report.json"]
            )
            result_reports = [_read(path) for path in result_paths if path.exists()]
            succeeded = any(value.get("passed") for value in result_reports)
            failure_text = (
                "\n".join(str(value.get("failure_reason") or value.get("error") or "") for value in result_reports)
                if result_reports
                else log.read_text(encoding="utf-8", errors="replace")
            )
            runtime_failure = not succeeded and any(marker in failure_text for marker in runtime_markers)
            if not runtime_failure or attempt == 3:
                return result.returncode
            archive = evidence / "runtime_retries" / f"{stage}_{attempt}"
            if not output_path.is_relative_to(evidence.resolve()) or not archive.resolve().is_relative_to(
                evidence.resolve()
            ):
                raise ValueError("runtime retry output must remain inside this route's evidence directory")
            archive.mkdir(parents=True)
            if output_path.exists():
                output_path.rename(archive / output_path.name)
            record["archived_output"] = str(archive)
            save_status()
            time.sleep(3)
        raise AssertionError("unreachable runtime retry state")

    def expect_passed(path: Path, stage: str, task: dict, world_hash: str | None = None) -> dict:
        value = _read(path)
        require_report_identity(value, task, label=stage, world_hash=world_hash)
        return value

    save_status()
    try:
        targets_path = evidence / "targets.json"
        invoke(
            "targets",
            "dry_run_curobo_pick_place_v2_targets_isaac.py",
            "--source-index",
            config.source_index,
            "--destination-index",
            config.destination_index,
            "--lift-clearance-m",
            config.lift_clearance_m,
            *routing_arguments(config.state4_routing),
            *source2_routing_arguments(config.source2_routing),
            "--output",
            targets_path,
            "--device",
            args.device,
            "--headless",
        )
        targets = _read(targets_path)
        task = targets["task_snapshot"]
        require_height(task, config.lift_clearance_m)
        if targets["config"]["lift_clearance_m"] != config.lift_clearance_m:
            raise ValueError("targets extractor did not apply the configured height")
        report.update(task_identity(task))
        require_report_identity(targets, task, label="targets")
        save_status()

        world_dir = evidence / "world"
        invoke(
            "world",
            "audit_curobo_v2_production_world_isaac.py",
            "--targets-report",
            targets_path,
            "--collision-model",
            args.collision_model,
            "--output",
            world_dir,
            "--device",
            args.device,
            "--headless",
        )
        world_path = world_dir / "report.json"
        world = _read(world_path)
        require_report_identity(world, task, label="world", require_passed=False)
        if not world.get("world_geometry_valid") or world["world_hash"] != world["world_evidence"]["world_hash"]:
            raise ValueError("route-specific production world geometry is invalid")
        world_hash = world["world_hash"]
        report["world_hash"] = world_hash
        save_status()

        pre_branches_path = evidence / "pre_pick_branches.json"
        invoke(
            "pre_pick_branches",
            "audit_curobo_v2_pre_pick_branches.py",
            "--world-report",
            world_path,
            "--targets-report",
            targets_path,
            "--waypoint",
            "pre_pick",
            "--seed-count",
            args.seed_count,
            "--output",
            pre_branches_path,
        )
        pre_branches = expect_passed(pre_branches_path, "pre-pick branches", task, world_hash)
        integrated = None
        integrated_dir = None
        report["artifact_failures"] = []
        for pick_number, pick_candidate in enumerate(
            (item for item in pre_branches["candidates"] if item["collision_free"]),
            start=1,
        ):
            if pick_number > args.max_candidates:
                break
            pick_suffix = f"{pick_number:02d}"
            first_dir = evidence / f"first_probe_{pick_suffix}"
            first_stage = f"first_probe_{pick_suffix}"
            invoke(
                first_stage,
                "probe_curobo_v2_noncontact.py",
                "--world-report",
                world_path,
                "--targets-report",
                targets_path,
                "--start-q",
                *targets["legacy_ik"]["q_start"],
                "--goal-q",
                *pick_candidate["joint_position"],
                "--segment",
                "free_space_to_pre_pick",
                "--disable-cuda-graph",
                "--graph-first",
                "--max-attempts",
                4,
                "--timeout-s",
                30,
                "--output",
                first_dir,
            )
            first_path = first_dir / "report.json"
            if not first_path.exists() or not _read(first_path).get("passed"):
                continue
            require_report_identity(_read(first_path), task, label=first_stage, world_hash=world_hash)

            contact_dir = evidence / f"contact_probe_{pick_suffix}"
            contact_stage = f"contact_probe_{pick_suffix}"
            invoke(
                contact_stage,
                "probe_curobo_v2_legacy_contact_isaac.py",
                "--first-probe",
                first_path,
                "--targets-report",
                targets_path,
                "--world-report",
                world_path,
                "--stop-after-lift",
                "--output",
                contact_dir,
                "--contact-policy",
                args.contact_policy,
                "--device",
                args.device,
                "--headless",
            )
            contact_path = contact_dir / "report.json"
            if not contact_path.exists() or not _read(contact_path).get("passed"):
                continue
            contact = _read(contact_path)
            require_report_identity(contact, task, label=contact_stage, world_hash=world_hash)

            branches_stage = f"place_high_branches_{pick_suffix}"
            branches_path = evidence / f"{branches_stage}.json"
            invoke(
                branches_stage,
                "audit_curobo_v2_pre_pick_branches.py",
                "--world-report",
                world_path,
                "--targets-report",
                targets_path,
                "--waypoint",
                "place_high",
                "--legacy-contact-report",
                contact_path,
                "--seed-count",
                args.seed_count,
                "--output",
                branches_path,
            )
            if not branches_path.exists():
                continue
            transfer_branches = _read(branches_path)
            require_report_identity(
                transfer_branches,
                task,
                label=branches_stage,
                require_passed=False,
                world_hash=world_hash,
            )
            if not transfer_branches.get("passed"):
                continue

            for place_number, place_candidate in enumerate(
                (item for item in transfer_branches["candidates"] if item["collision_free"]),
                start=1,
            ):
                if place_number > args.max_candidates:
                    break
                suffix = f"{pick_number:02d}_{place_number:02d}"
                transfer_stage = f"transfer_probe_{suffix}"
                transfer_dir = evidence / transfer_stage
                invoke(
                    transfer_stage,
                    "probe_curobo_v2_noncontact.py",
                    "--world-report",
                    world_path,
                    "--targets-report",
                    targets_path,
                    "--legacy-contact-report",
                    contact_path,
                    "--start-q",
                    *contact["lift_q"],
                    "--goal-q",
                    *place_candidate["joint_position"],
                    "--segment",
                    "free_space_transfer",
                    "--disable-cuda-graph",
                    "--graph-first",
                    "--max-attempts",
                    4,
                    "--timeout-s",
                    30,
                    "--output",
                    transfer_dir,
                )
                transfer_path = transfer_dir / "report.json"
                if not transfer_path.exists() or not _read(transfer_path).get("passed"):
                    continue
                require_report_identity(_read(transfer_path), task, label=transfer_stage, world_hash=world_hash)

                integrated_stage = f"integrated_{suffix}"
                candidate_integrated_dir = evidence / integrated_stage
                invoke(
                    integrated_stage,
                    "probe_curobo_v2_legacy_contact_isaac.py",
                    "--first-probe",
                    first_path,
                    "--targets-report",
                    targets_path,
                    "--world-report",
                    world_path,
                    "--place-high-q",
                    *place_candidate["joint_position"],
                    "--integrated",
                    "--output",
                    candidate_integrated_dir,
                    "--contact-policy",
                    args.contact_policy,
                    "--device",
                    args.device,
                    "--headless",
                )
                integrated_path = candidate_integrated_dir / "report.json"
                if not integrated_path.exists() or not _read(integrated_path).get("passed"):
                    continue
                require_report_identity(_read(integrated_path), task, label=integrated_stage, world_hash=world_hash)
                artifact_stage = f"artifact_{suffix}"
                candidate_artifact_dir = evidence / artifact_stage
                artifact_exit = invoke(
                    artifact_stage,
                    "build_curobo_pick_place_v2_artifact.py",
                    "--integrated-run",
                    candidate_integrated_dir,
                    "--targets-report",
                    targets_path,
                    "--world-report",
                    world_path,
                    "--output",
                    candidate_artifact_dir,
                    "--contact-policy",
                    args.contact_policy,
                )
                if artifact_exit != 0:
                    artifact_report = _read(candidate_artifact_dir / "plan_report.json")
                    report["artifact_failures"].append(
                        {"candidate": suffix, "reason": artifact_report.get("failure_reason")}
                    )
                    save_status()
                    continue
                try:
                    candidate_artifact = verify_hybrid_artifact(candidate_artifact_dir)
                    require_report_identity(
                        candidate_artifact["report"],
                        task,
                        label=artifact_stage,
                        world_hash=world_hash,
                    )
                except Exception as error:
                    report["artifact_failures"].append(
                        {"candidate": suffix, "reason": f"{type(error).__name__}: {error}"}
                    )
                    save_status()
                    continue
                shutil.move(str(candidate_artifact_dir), str(output))
                integrated, integrated_dir = _read(integrated_path), candidate_integrated_dir
                report["first_probe"] = str(first_path)
                report["contact_probe"] = str(contact_path)
                report["place_high_branches"] = str(branches_path)
                report["transfer_probe"] = str(transfer_path)
                break
            if integrated is not None:
                break
        if integrated is None:
            raise RuntimeError(
                "no candidate combination passed CuRobo planning, Isaac integration, and hybrid trajectory acceptance; "
                + str(report["artifact_failures"])
            )
        report["integrated_run"] = str(integrated_dir)
        save_status()

        verified = verify_hybrid_artifact(output)
        require_report_identity(verified["report"], task, label="hybrid artifact", world_hash=world_hash)
        report["passed"] = True
        report["stage"] = "complete"
        report["samples"] = verified["acceptance"]["samples"]
        save_status()
        print(json.dumps({"passed": True, "run": str(output), "evidence": str(evidence), "samples": report["samples"]}))
        return 0
    except Exception as error:
        report["passed"] = False
        report["failure_reason"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        save_status()
        print(
            json.dumps(
                {
                    "passed": False,
                    "stage": report["stage"],
                    "failure_reason": report["failure_reason"],
                    "evidence": str(evidence),
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
