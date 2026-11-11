"""Audit live GUI contact captures against their measured trace and make a preview."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    report = read(run / "report.json")
    visual = run / "execution_visualization"
    manifest = read(visual / "frames.json")
    trace_path = Path(manifest["source_trace"])
    assert trace_path.resolve() == (run / "trace.json").resolve()
    assert hashlib.sha256(trace_path.read_bytes()).hexdigest() == manifest["source_trace_sha256"]
    trace = read(trace_path)
    records = trace["records"] if isinstance(trace, dict) else trace
    assert report["mode"] == "gui" and manifest["execution_passed"] == report["passed"]
    assert manifest["execution_contract"] == report["execution_contract"]
    assert manifest["source_frames"] == manifest["render_calls"] == len(records)
    cameras = ("HandCamera", "CameraSensor_2", "CameraSensor_3")
    frames = manifest["frames"]
    assert len(frames) == len(records) // 12 * 3 and frames
    seen = set()
    image_sha256 = {}
    for frame in frames:
        index = frame["trajectory_index"]
        record = records[index]
        key = (index, frame["camera"])
        assert key not in seen and key[1] in cameras
        seen.add(key)
        assert frame["render_frame"] == index + 1 and (index + 1) % 12 == 0
        assert abs(frame["time_s"] - index / report["execution_contract"]["control_hz"]) < 1e-12
        assert frame["execution_stage"] == record["stage"]
        if "contact_waypoint" in frame:
            assert frame["contact_waypoint"] == record.get("contact_waypoint")
        for field, source in (
            ("joint_position", "q"),
            ("joint_velocity", "dq"),
            ("observed_target_pose_w", "target_pose_w"),
        ):
            assert frame[field] == record[source] and np.all(np.isfinite(frame[field]))
        path = (visual / frame["rgb"]).resolve()
        assert path.is_relative_to(visual.resolve())
        with Image.open(path) as image:
            assert image.size == (640, 480)
            image.verify()
        image_sha256[frame["rgb"]] = hashlib.sha256(path.read_bytes()).hexdigest()
    indices = sorted({frame["trajectory_index"] for frame in frames})
    assert all((index, camera) in seen for index in indices for camera in cameras)
    chosen = [indices[round(fraction * (len(indices) - 1))] for fraction in (0, 0.3, 0.65, 1)]
    sheet = Image.new("RGB", (996, 1140), "#eeeeee")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
    draw.text((12, 8), f"{run.name} | live GUI | integration passed={report['passed']}", font=font, fill="black")
    selected = []
    for row, index in enumerate(chosen):
        y = 44 + row * 272
        label = f"{index}: {records[index]['stage']} / {records[index].get('contact_waypoint') or ''}"
        draw.text((12, y), label, font=font, fill="black")
        for col, camera in enumerate(cameras):
            frame = next(f for f in frames if f["trajectory_index"] == index and f["camera"] == camera)
            with Image.open(visual / frame["rgb"]) as image:
                sheet.paste(image.convert("RGB").resize((320, 240), Image.Resampling.LANCZOS), (12 + col * 328, y + 26))
            selected.append(dict(camera=camera, trajectory_index=index, rgb=str(visual / frame["rgb"])))
    preview = run / "live_gui_preview.jpg"
    sheet.save(preview, quality=94)
    audit = dict(
        capture_audit_passed=True,
        scope=(
            "live mixed execution: state replay for free space, physics for contacts; "
            "capture audit is not trajectory acceptance"
        ),
        integration_passed=report["passed"],
        failure_reason=report.get("failure_reason"),
        source_frames=len(records),
        image_count=len(frames),
        resolution=[640, 480],
        execution_contract=report["execution_contract"],
        stage_counts={
            stage: sum(r["stage"] == stage for r in records) for stage in dict.fromkeys(r["stage"] for r in records)
        },
        contact_waypoint_counts={
            name: sum(r.get("contact_waypoint") == name for r in records)
            for name in dict.fromkeys(r.get("contact_waypoint") for r in records)
            if name is not None
        },
        image_sha256=image_sha256,
        preview=str(preview),
        inspection_samples=selected,
    )
    (run / "capture_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in audit.items() if k not in ("inspection_samples", "image_sha256")}))


if __name__ == "__main__":
    main()
