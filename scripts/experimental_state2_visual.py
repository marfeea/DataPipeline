"""Isolated 2->x visual experiments: ignore self/velocity rejection, save every frame."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))
from state2_relaxed_visual_worker import POLICY


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_complete(run, targets_path, world_path):
    import numpy as np
    from DataPipe.pipelines.curobo_bootstrap.pick_place_artifact import (
        assemble_hybrid_trajectory, save_hybrid_trajectory, validate_hybrid_trajectory,
    )
    from DataPipe.pipelines.curobo_bootstrap.planner import load_trajectory, read_urdf_joint_limits

    report = read(run / "report.json")
    if not report["passed"] or report["stage"] != "complete":
        raise ValueError("Only a completed grab/transfer/place/release may be exported as complete")
    trace = read(run / "trace.json")
    arrays = assemble_hybrid_trajectory(trace, load_trajectory(run / "first_trajectory.npz"),
                                       load_trajectory(run / "transfer_trajectory.npz"))
    model = read(read(world_path)["collision_model"])
    urdf = Path(model["urdf"])
    _, _, limits = read_urdf_joint_limits(urdf)
    strict = dict(passed=False)
    try:
        strict = validate_hybrid_trajectory(arrays, urdf=urdf)
    except ValueError as error:
        strict["failure_reason"] = str(error)
    legacy = arrays["segment_source"] != "curobo"
    metadata = dict(
        schema="state2_relaxed_complete_trajectory/1", diagnostic_policy=POLICY,
        formal_accepted=False, execution_complete=True, samples=len(trace),
        source_trace_sha256=digest(run / "trace.json"), strict_hybrid_gate=strict,
        velocity_gate_passed=bool(np.all(np.abs(arrays["joint_velocity"]) <= limits + 1e-5)),
        peak_joint_velocity_rad_s=np.max(np.abs(arrays["joint_velocity"]), axis=0).tolist(),
        peak_legacy_acceleration_rad_s2=float(np.max(np.abs(arrays["joint_acceleration"][legacy]))),
        legacy_acceleration_gate_passed=bool(np.all(np.abs(arrays["joint_acceleration"][legacy]) <= 50)),
        first_segment_strict_passed=report["curobo_first"]["acceptance"]["strict_passed"],
        transfer_segment_strict_passed=report["curobo_transfer"]["acceptance"]["strict_passed"],
        max_first_self_collision_score=report["curobo_first"]["acceptance"]["metrics"]["max_self_collision_score"],
        max_transfer_self_collision_score=report["curobo_transfer"]["acceptance"]["metrics"]["max_self_collision_score"],
        final_target_pose_w=trace[-1]["target_pose_w"],
    )
    save_hybrid_trajectory(run / "diagnostic_trajectory.npz", arrays)
    metadata["diagnostic_trajectory_sha256"] = digest(run / "diagnostic_trajectory.npz")
    write(run / "diagnostic_trajectory_report.json", metadata)
    return metadata


def make_videos(run, route):
    import imageio_ffmpeg
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    report = read(run / "report.json")
    trace = read(run / "trace.json")
    visual = run / "execution_visualization"
    frames = read(visual / "frames.json")
    names = ("HandCamera", "CameraSensor_2", "CameraSensor_3")
    n = len(trace)
    if not n or len(frames["frames"]) != n * 3 or frames["frame_interval_render_calls"] != 1:
        raise ValueError("Dense native image capture is incomplete")
    indexed = {(f["trajectory_index"], f["camera"]): f for f in frames["frames"]}
    if len(indexed) != n * 3:
        raise ValueError("Duplicate camera/frame identity")
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
    outputs = {name: run / (name + ".mp4") for name in names}
    outputs["three_cameras"] = run / "three_cameras.mp4"
    writers = {}
    try:
        for name, path in outputs.items():
            if path.exists():
                raise FileExistsError(path)
            size = (1920, 512) if name == "three_cameras" else (640, 480)
            writer = imageio_ffmpeg.write_frames(str(path), size, fps=60, codec="libx264",
                macro_block_size=2, output_params=["-preset", "veryfast", "-crf", "20", "-movflags", "+faststart", "-threads", "4"])
            writer.send(None)
            writers[name] = writer
        for index, sample in enumerate(trace):
            panel = Image.new("RGB", (1920, 512), "black")
            ImageDraw.Draw(panel).text((8, 5),
                f"DIAGNOSTIC {route} | self/velocity gates relaxed | {index / 60:.3f}s | {sample['stage']}",
                font=font, fill="white")
            for camera_index, name in enumerate(names):
                frame = indexed[index, name]
                if frame["render_frame"] != index + 1 or frame["execution_stage"] != sample["stage"]:
                    raise ValueError("Image stage/frame binding mismatch")
                if not np.allclose(frame["joint_position"], sample["q"], rtol=0, atol=1e-7):
                    raise ValueError("Image joint state does not match native trace")
                if not np.allclose(frame["joint_velocity"], sample["dq"], rtol=0, atol=1e-7):
                    raise ValueError("Image velocity does not match native trace")
                with Image.open(visual / frame["rgb"]) as source:
                    source.load()
                    if source.size != (640, 480):
                        raise ValueError("Unexpected native camera resolution")
                    rgb = source.convert("RGB")
                writers[name].send(np.asarray(rgb))
                panel.paste(rgb, (camera_index * 640, 32))
            writers["three_cameras"].send(np.asarray(panel))
    finally:
        for writer in writers.values():
            writer.close()
    summaries = []
    for name, path in outputs.items():
        count, seconds = imageio_ffmpeg.count_frames_and_secs(str(path))
        if count != n or abs(seconds - n / 60) > 0.05:
            raise ValueError(f"Video frame count/time mismatch: {name}, {count}, {seconds}")
        summaries.append(dict(camera=name, path=str(path), frames=count, duration_s=seconds, sha256=digest(path)))
    audit = dict(schema="state2_dense_video_audit/1", passed=True, diagnostic_only=True,
                 execution_complete=report["passed"], samples=n, image_count=n * 3,
                 first_stage=trace[0]["stage"], last_stage=trace[-1]["stage"],
                 stages=list(dict.fromkeys(r["stage"] for r in trace)), videos=summaries)
    write(run / "video_audit.json", audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--destinations", type=int, nargs="+", choices=(0, 1, 3, 4), default=[0, 1, 3])
    parser.add_argument("--max-candidates", type=int, default=4)
    args = parser.parse_args()
    if len(set(args.destinations)) != len(args.destinations) or not 1 <= args.max_candidates <= 8:
        parser.error("unique destinations and max-candidates in [1,8] required")
    batch = args.output_root.resolve()
    if batch.exists():
        raise FileExistsError("Use a new output-root; existing evidence will not be overwritten")
    batch.mkdir(parents=True)
    runtime = Path(sys.executable).parent
    os.environ["PATH"] = os.pathsep.join([str(runtime), str(runtime / "Scripts"), str(runtime / "Library/bin"), os.environ.get("PATH", "")])
    source_files = [PROJECT / "scripts" / f for f in (
        "dry_run_curobo_pick_place_v2_targets_isaac.py", "audit_curobo_v2_production_world_isaac.py",
        "audit_curobo_v2_pre_pick_branches.py", "probe_curobo_v2_noncontact.py",
        "probe_curobo_v2_legacy_contact_isaac.py", "experimental_state2_visual.py", "state2_relaxed_visual_worker.py")]
    source_files += list((PROJECT / "DataPipe/pipelines/curobo_bootstrap").glob("*.py"))
    source_files += [PROJECT / "DataPipe/source2_routing.py", PROJECT / "DataPipe/sequence.py", PROJECT / "configs/task.py",
                     PROJECT / "DataPipe/collection_cameras.py"]
    frozen = {str(p): digest(p) for p in source_files}
    model = PROJECT / "data/curobo_pick_place_graph_run003_20261003_r2/collision_model.json"
    frozen[str(model)] = digest(model)
    write(batch / "experiment_manifest.json", dict(schema="state2_relaxed_visual_batch/1", diagnostic_policy=POLICY,
          destinations=args.destinations, source_sha256=frozen, collision_model_sha256=digest(model),
          parameters=dict(side_m=0.06, pre_pick_lift_m=0.04, lift_clearance_m=0.3, physics_hz=120, control_hz=60)))
    records = []

    def verify():
        for filename, expected in frozen.items():
            if digest(filename) != expected:
                raise ValueError(f"Original/frozen source changed: {filename}")

    def save():
        write(batch / "collection_result.json", dict(schema="state2_relaxed_visual_result/1", diagnostic_only=True,
              formal_accepted=False, records=records, finished=len(records) == len(args.destinations) and all(r.get("finished") for r in records),
              original_sources_unchanged=True))

    def invoke(stage, script, arguments, output, kind=None):
        verify()
        if kind:
            command = [sys.executable, str(PROJECT / "scripts/state2_relaxed_visual_worker.py"), "--kind", kind, "--", *map(str, arguments)]
        else:
            command = [sys.executable, str(PROJECT / "scripts" / script), *map(str, arguments)]
        print("STATE2_VISUAL_STAGE " + json.dumps(dict(route=route, stage=stage)), flush=True)
        with (evidence / (stage + ".log")).open("w", encoding="utf-8") as log:
            code = subprocess.run(command, cwd=PROJECT, env=dict(os.environ, HEADLESS="0"),
                                  stdout=log, stderr=subprocess.STDOUT, check=False).returncode
        value_path = output if output.suffix == ".json" else output / "report.json"
        value = read(value_path) if value_path.exists() else dict(passed=False, failure_reason=f"No report; exit={code}")
        record["attempts"].append(dict(stage=stage, returncode=code, passed=value.get("passed"), failure_reason=value.get("failure_reason") or value.get("error")))
        save()
        return value

    for destination in args.destinations:
        route = f"2_to_{destination}"
        evidence = batch / route
        evidence.mkdir()
        record = dict(route=route, visual_complete=False, finished=False, attempts=[])
        records.append(record)
        save()
        try:
            targets_path = evidence / "targets.json"
            routing = ["--state2-pre-pick-side-m", 0.06, "--state2-pre-pick-lift-m", 0.04]
            if destination == 4:
                routing += ["--state4-routing", "--state4-place-standoff-m", 0.24,
                            "--state4-place-entry-lift-m", 0.012, "--state4-place-measured-grasp"]
            targets = invoke("targets", "dry_run_curobo_pick_place_v2_targets_isaac.py",
                ["--source-index", 2, "--destination-index", destination, "--lift-clearance-m", 0.3,
                 *routing, "--device", "cuda:0", "--output", targets_path], targets_path)
            if not targets["passed"]:
                raise RuntimeError("Target IK failed")
            world_path = evidence / "world/report.json"
            world = invoke("world", "audit_curobo_v2_production_world_isaac.py",
                ["--targets-report", targets_path, "--collision-model", model, "--device", "cuda:0", "--output", world_path.parent], world_path.parent)
            if not world.get("world_geometry_valid"):
                raise RuntimeError("Production world geometry failed")
            common = ["--targets-report", targets_path, "--world-report", world_path]
            branches_path = evidence / "pre_pick_branches.json"
            branches = invoke("pre_pick_branches", "", [*common, "--seed-count", 200, "--output", branches_path], branches_path, "branches")
            record["pre_pick"] = {k: branches[k] for k in ("candidate_count", "collision_free_count", "strict_passed")}
            for pick_number, candidate in enumerate((c for c in branches["candidates"] if c["collision_free"]), 1):
                if pick_number > args.max_candidates:
                    break
                first_dir = evidence / f"first_{pick_number:02d}"
                first = invoke(first_dir.name, "", [*common, "--start-q", *targets["legacy_ik"]["q_start"],
                    "--goal-q", *candidate["joint_position"], "--segment", "free_space_to_pre_pick",
                    "--disable-cuda-graph", "--graph-first", "--max-attempts", 4, "--timeout-s", 30,
                    "--output", first_dir], first_dir, "noncontact")
                if not first["passed"]:
                    continue
                contact_dir = evidence / f"pick_lift_{pick_number:02d}"
                contact_args = [*common, "--first-probe", first_dir / "report.json", "--contact-policy", "run003",
                                "--physics-hz", 120, "--device", "cuda:0", "--capture-execution"]
                contact = invoke(contact_dir.name, "", [*contact_args, "--stop-after-lift", "--output", contact_dir], contact_dir, "contact")
                if not contact["passed"]:
                    continue
                place_path = evidence / f"place_branches_{pick_number:02d}.json"
                place = invoke(place_path.stem, "", [*common, "--waypoint", "place_high", "--legacy-contact-report",
                    contact_dir / "report.json", "--seed-count", 200, "--output", place_path], place_path, "branches")
                if not place["passed"]:
                    continue
                for place_number, place_candidate in enumerate((c for c in place["candidates"] if c["collision_free"]), 1):
                    if place_number > args.max_candidates:
                        break
                    integrated_dir = evidence / f"integrated_{pick_number:02d}_{place_number:02d}"
                    integrated = invoke(integrated_dir.name, "", [*contact_args, "--integrated", "--place-high-q",
                        *place_candidate["joint_position"], "--output", integrated_dir], integrated_dir, "contact")
                    if not integrated["passed"]:
                        continue
                    metadata = export_complete(integrated_dir, targets_path, world_path)
                    video = make_videos(integrated_dir, route)
                    record.update(visual_complete=True, selected_run=str(integrated_dir),
                                  trajectory=metadata, video_audit=video,
                                  selected_pre_pick_self_collision_score=candidate["self_collision_score"])
                    break
                if record["visual_complete"]:
                    break
            if not record["visual_complete"]:
                record["failure_reason"] = "No candidate completed grab/transfer/place/release with world and IK gates retained"
        except Exception as error:
            import traceback
            record["failure_reason"] = f"{type(error).__name__}: {error}"
            record["traceback"] = traceback.format_exc()
        verify()
        record["finished"] = True
        save()
        print("STATE2_VISUAL_RESULT " + json.dumps(record), flush=True)
    return 0 if all(r["visual_complete"] for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
