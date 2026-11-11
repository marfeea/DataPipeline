"""Independently verify artifacts, three-view captures, and fixed-view scene visibility."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

import numpy as np
from PIL import Image

from testscripts.audit_run003_isaacsim import inspect_frames
from scripts.pipelines.curobo_bootstrap.pick_place_artifact import verify_hybrid_artifact


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--routes", nargs="*", help="Audit selected completed routes and retain prior results")
    args = parser.parse_args()
    batch = args.output_root.resolve()
    summary = read(batch / "graph_summary.json")
    reference = batch / "4_to_2_evidence/integrated_01_01/execution_visualization/camera3/CameraSensor_3_render_frame_0012_rgb.png"
    with Image.open(reference) as image:
        template = np.asarray(image, dtype=float)[:50].copy()
    previous = batch / "independent_audit.json"
    results = read(previous)["routes"] if args.routes and previous.exists() else {}
    for name, route in summary["routes"].items():
        if args.routes and name not in args.routes:
            results.setdefault(name, dict(passed=False, failure_reason="route not yet complete"))
            continue
        result = dict(passed=False)
        results[name] = result
        if not route.get("passed"):
            result["failure_reason"] = route.get("failure_reason", "route not yet complete")
            continue
        try:
            run = Path(route["run"])
            integrated = Path(route["integrated_run"])
            artifact = verify_hybrid_artifact(run)
            source, destination = map(int, name.split("_to_"))
            task = artifact["snapshot"]["task"]
            assert (task["source_index"], task["destination_index"]) == (source, destination)
            collection = read(run.with_name(run.name + "_evidence") / "run_report.json")
            assert collection["passed"] and collection["mode"] == "gui" and collection["backend"] == "isaacsim"
            assert collection["capture_execution"]
            assert all("--headless" not in stage["command"] and "--no-gui" not in stage["command"] for stage in collection["stages"])
            report = read(integrated / "report.json")
            assert report["passed"] and report["mode"] == "gui" and report["backend"] == "isaacsim"
            trace = read(integrated / "trace.json")
            records = trace["records"] if isinstance(trace, dict) else trace
            replay = read(run / "replay_report.json")
            assert replay["passed"] and replay["mode"] == "gui" and replay["backend"] == "isaacsim"
            captures = []
            for directory, count, measured in (
                (integrated / "execution_visualization", len(records), records),
                (Path(route["replay_visualization"]), len(artifact["trajectory"]["time_s"]), None),
            ):
                audit = inspect_frames(directory, source_count=count, trace=measured)
                invalid = []
                for frame in read(directory / "frames.json")["frames"]:
                    if frame["camera"] != "CameraSensor_3":
                        continue
                    with Image.open(directory / frame["rgb"]) as image:
                        difference = float(np.abs(np.asarray(image, dtype=float)[:50] - template).mean())
                    # The fixed front view's ceiling remains visible above
                    # robot motion. Background-only renders lose this anchor.
                    if difference > 20:
                        invalid.append(dict(rgb=frame["rgb"], ceiling_difference=difference))
                audit.update(directory=str(directory), scene_visibility_passed=not invalid, invalid_frames=invalid)
                captures.append(audit)
            result["captures"] = captures
            assert all(c["scene_visibility_passed"] for c in captures), "fixed-view scene geometry missing in captured RGB"
            result.update(passed=True, samples=len(artifact["trajectory"]["time_s"]))
        except Exception as error:
            result["failure_reason"] = f"{type(error).__name__}: {error}"
    audit = dict(schema="five_state_gui_independent_audit/1", passed=len(results) == 20 and all(r["passed"] for r in results.values()), successful_routes=sum(r["passed"] for r in results.values()), routes=results)
    (batch / "independent_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    lines = ["# 五状态双向轨迹 GUI 测试", "", f"独立验收通过：{audit['successful_routes']}/20。后端 Isaac Sim；全部使用 GUI。", "", "| 起点→终点 | 0 | 1 | 2 | 3 | 4 |", "| --- | --- | --- | --- | --- | --- |"]
    for source in range(5):
        row = [str(source)]
        for destination in range(5):
            name = f"{source}_to_{destination}"
            if source == destination:
                row.append("—")
            elif results.get(name, {}).get("passed"):
                row.append(f"[通过]({Path(summary['routes'][name]['run']).name}/trajectory.npz)")
            elif name not in results or results[name].get("failure_reason") == "route not yet complete":
                row.append("待完成")
            else:
                row.append("未通过")
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "成功条件：轨迹验收、GUI 执行与回放、三路图像解码、帧索引和实测状态绑定、渲染帧刷新以及固定视角场景可见性全部通过。", "", "沿用示例的 run003 混合规划/接触执行口径；托盘接触为诊断项。", "", "| 路线 | 样本 | 现场图片 | 回放图片 | 状态或原因 |", "| --- | ---: | ---: | ---: | --- |"]
    for name, result in results.items():
        record = summary["routes"][name]
        reason = "通过" if result["passed"] else result.get("failure_reason", "未通过")
        lines.append(f"| {name} | {record.get('samples', '—')} | {record.get('native_images', '—')} | {record.get('replay_images', '—')} | {reason.replace('|', '/')} |")
    (batch / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({name: {k:v for k,v in result.items() if k != "captures"} for name,result in results.items() if not args.routes or name in args.routes}, indent=2))
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
