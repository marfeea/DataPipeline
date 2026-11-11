"""Audit newly rendered periodic images and build a local three-view replay gallery."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
CAMERAS = ["HandCamera", "CameraSensor_2", "CameraSensor_3"]
STAGES = {"legacy_approach_pick": "接近", "close_gripper": "夹紧", "grasp_settle": "夹持稳定",
          "legacy_lift": "抬升", "legacy_lift_hold": "抬升后保持"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def font(size):
    return ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", size)


def make_sheet(root, label, frames):
    # Every native image remains untouched. Labels are added only to this derived sheet.
    selected = [f for f in frames if f["camera"] == "CameraSensor_3" and f["control_step"] >= 240]
    width, cell, caption, header = 320, 320, 55, 76
    sheet = Image.new("RGB", (width * 4, header + ((len(selected) + 3) // 4) * (cell + caption)), "#f3f5f8")
    draw = ImageDraw.Draw(sheet)
    draw.text((15, 8), f"{label} 实测轨迹重播：夹持、抬升和保持", font=font(25), fill="#172635")
    draw.text((15, 44), "每 12 个控制步一张 / 0.2 秒间隔 / CameraSensor_3", font=font(18), fill="#48596a")
    for index, frame in enumerate(selected):
        x, y = index % 4 * width, header + index // 4 * (cell + caption)
        with Image.open(root / label / "execution_visualization" / frame["rgb"]) as image:
            sheet.paste(image.convert("RGB").resize((cell, cell), Image.Resampling.LANCZOS), (x, y))
        draw.text((x + 7, y + cell + 3), f"第 {frame['control_step']} 步 · {frame['time_s']:.2f} s", font=font(18), fill="#172635")
        draw.text((x + 7, y + cell + 28), STAGES[frame["execution_stage"]], font=font(17), fill="#48596a")
    path = root / label / "lift_overview.png"
    sheet.save(path)
    # Eight native views make a compact overview for inline display.
    compact_frames = [next(f for f in frames if f["camera"] == camera and f["control_step"] == step)
                      for step in (240, 288, 336, 384) for camera in ("CameraSensor_2", "CameraSensor_3")]
    compact = Image.new("RGB", (384 * 2, 60 + 4 * 425), "#f3f5f8")
    draw = ImageDraw.Draw(compact)
    draw.text((10, 10), f"{label} 重播选帧 · 双外部相机", font=font(24), fill="#172635")
    for index, frame in enumerate(compact_frames):
        x, y = index % 2 * 384, 60 + index // 2 * 425
        with Image.open(root / label / "execution_visualization" / frame["rgb"]) as image:
            compact.paste(image.convert("RGB"), (x, y))
        draw.text((x + 6, y + 389), f"{frame['control_step']} 步 · {frame['time_s']:.2f}s · {STAGES[frame['execution_stage']]}",
                  font=font(19), fill="#172635")
    compact.save(root / label / "selected_views.png")
    animation = []
    for frame in frames:
        if frame["camera"] != "CameraSensor_3":
            continue
        view = Image.new("RGB", (384, 430), "#f3f5f8")
        with Image.open(root / label / "execution_visualization" / frame["rgb"]) as image:
            view.paste(image.convert("RGB"), (0, 0))
        draw = ImageDraw.Draw(view)
        draw.text((7, 387), f"第 {frame['control_step']} 步 · {frame['time_s']:.2f}s", font=font(17), fill="#172635")
        draw.text((7, 409), STAGES[frame["execution_stage"]], font=font(15), fill="#48596a")
        animation.append(view.convert("P", palette=Image.Palette.ADAPTIVE, colors=256))
    animation[0].save(root / label / "replay_camera3.gif", save_all=True, append_images=animation[1:],
                      duration=200, loop=0, disposal=2, optimize=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="legacy_timed_lift_replay_20261021")
    args = parser.parse_args()
    root = PROJECT / "data/test/trajdata" / args.run_id / "0_to_1"
    summary = read(root / "summary.json")
    assert summary["passed"] and summary["all_source_files_unchanged"]
    audit = dict(passed=False, image_count=0, capture_interval_control_steps=12, trials=[], images=[])
    gallery = []
    for trial in summary["trials"]:
        label = trial["trial"]
        assert trial["passed"] and trial["full_trajectory_replayed"] and trial["physics_steps"] == 0
        manifest = read(root / label / "execution_visualization/frames.json")
        frames = manifest["frames"]
        expected = list(range(12, trial["source_frames"] + 1, 12))
        source = PROJECT / "data/test/trajectorys/legacy_timed_lift_20261021_verified/0_to_1" / label
        source_trace = read(source / "trace.json")
        expected_steps = [(step, camera) for step in expected for camera in CAMERAS]
        assert [(f["control_step"], f["camera"]) for f in frames] == expected_steps
        assert len(frames) == 102 and manifest["render_calls"] == 411
        by_camera = {camera: [] for camera in CAMERAS}
        for frame in frames:
            step = frame["control_step"]
            assert frame["trajectory_index"] == step - 1
            assert abs(frame["time_s"] - (step - 1) / 60) < 1e-12
            assert frame["execution_stage"] == source_trace[step - 1]["stage"]
            assert frame.get("sensor_render_reference") is not None
            image_path = root / label / "execution_visualization" / frame["rgb"]
            with Image.open(image_path) as image:
                image.load()
                assert image.size == (384, 384) and image.mode == "RGB"
                pixels = np.asarray(image)
                std = float(pixels.std())
                assert std > 5, image_path
            digest = sha(image_path)
            by_camera[frame["camera"]].append(digest)
            audit["images"].append(dict(trial=label, control_step=step, camera=frame["camera"],
                                        image=str(image_path), sha256=digest, rgb_std=std))
        for camera in CAMERAS:
            assert len(set(by_camera[camera])) > 1, f"stale camera: {camera}"
        make_sheet(root, label, frames)
        audit["trials"].append(dict(trial=label, groups=len(expected), images=len(frames),
                                     first_step=expected[0], last_step=expected[-1],
                                     unique_images_by_camera={c: len(set(v)) for c, v in by_camera.items()}))
        audit["image_count"] += len(frames)
        gallery.append(dict(label=label, frames=[dict(step=step, time=(step - 1) / 60,
                           stage=STAGES[source_trace[step - 1]["stage"]],
                           images=[dict(camera=camera, path=f"{label}/execution_visualization/" +
                              next(f["rgb"] for f in frames if f["control_step"] == step and f["camera"] == camera))
                                   for camera in CAMERAS]) for step in expected]))
    audit["manual_visual_review"] = dict(
        action_evidence_cameras=["CameraSensor_2", "CameraSensor_3"],
        robot_and_lifted_object_visible=True,
        hand_camera_observation="Reviewed grasp/lift samples face the outdoor background; retained unchanged, not used to verify the lift",
        reviewed_examples=["trial_01/selected_views.png", "trial_01/lift_overview.png",
                           "trial_02/execution_visualization/handCamera/HandCamera_render_frame_0336_rgb.png",
                           "trial_03/execution_visualization/camera3/CameraSensor_3_render_frame_0408_rgb.png"])
    audit["passed"] = True
    (root / "visual_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    data = json.dumps(gallery, ensure_ascii=False).replace("<", "\\u003c")
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>实测轨迹重播 · 每12步图像</title><style>
body{font:16px system-ui,"Microsoft YaHei",sans-serif;margin:0;background:#eef2f6;color:#182736}main{max-width:1320px;margin:auto;padding:28px}
h1{font-size:28px}p{line-height:1.7}.bar{display:flex;align-items:center;gap:15px;flex-wrap:wrap;background:white;padding:16px;border-radius:14px}
button,select{font:inherit;padding:8px 14px;border:1px solid #b9c4d0;border-radius:7px;background:white;cursor:pointer}input{flex:1;min-width:200px}
.views{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:20px}figure{margin:0;background:white;border-radius:12px;padding:10px}
img{width:100%;display:block}figcaption{padding:8px}a{color:#245cd0}.status{font-size:20px;margin:18px 0}footer{padding:22px 0}
@media(max-width:850px){.views{grid-template-columns:1fr}}
</style><main><h1>实测轨迹重播</h1><p>三条验收通过的实测记录。每 12 个控制步保存三路原生相机图像，采样间隔 0.2 秒。<br>显示已记录的机械臂、手指及物体状态；本次不重新规划或进行动力学测试。</p>
<p>动作观察请看外部相机 CameraSensor_2 / CameraSensor_3。手眼相机在已检查的抓取和抬升帧中朝向室外背景，已原样保留。</p>
<div class="bar"><select id="trial"></select><button id="prev">上一组</button><button id="play">播放</button><button id="next">下一组</button><input id="seek" type="range" min="0" max="33" value="0"></div>
<div id="status" class="status"></div><div id="views" class="views"></div><footer><a id="overview">查看连续抬升图像总览</a> · <a href="visual_audit.json">图像核验及SHA256</a> · <a href="summary.json">重播报告</a></footer></main>
<script>const data=DATA;let index=0,timer=null;const trial=document.getElementById('trial'),seek=document.getElementById('seek');
for(let i=0;i<data.length;i++){let o=document.createElement('option');o.value=i;o.textContent=data[i].label;trial.append(o)}
function show(){const t=data[+trial.value],f=t.frames[index];seek.value=index;document.getElementById('status').textContent=`第 ${f.step} / 411 控制步 · ${f.time.toFixed(2)} 秒 · ${f.stage} · 第 ${index+1} / ${t.frames.length} 组`;
document.getElementById('views').innerHTML=f.images.map(v=>`<figure><a href="${v.path}" target="_blank"><img src="${v.path}" alt="${v.camera}"></a><figcaption>${v.camera}</figcaption></figure>`).join('');document.getElementById('overview').href=t.label+'/lift_overview.png'}
trial.onchange=()=>{index=0;show()};seek.oninput=()=>{index=+seek.value;show()};document.getElementById('prev').onclick=()=>{index=Math.max(0,index-1);show()};document.getElementById('next').onclick=()=>{index=Math.min(33,index+1);show()};
document.getElementById('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;document.getElementById('play').textContent='播放'}else{document.getElementById('play').textContent='暂停';timer=setInterval(()=>{index=(index+1)%34;show()},200)}};show();</script></html>'''.replace("DATA", data)
    (root / "index.html").write_text(page, encoding="utf-8")
    shutil.copy2(__file__, root / "gallery_source.py")
    shutil.copy2(HERE / "README.md", root / "README.md")
    shutil.copy2(HERE / "replay.log", root / "replay.log")
    report = "# 实测轨迹重播图像\n\n三条轨迹均在可见的 IsaacSim GUI 中完整重播。严格每 12 个控制步捕获一次，每组包含三路原生相机。\n\n"
    report += "| 记录 | 控制步 | 图像组 | 原生 PNG | 模式 |\n|---|---:|---:|---:|---|\n"
    for trial in audit["trials"]:
        report += f"| {trial['trial']} | 411 | {trial['groups']} | {trial['images']} | GUI 实测状态重播 |\n"
    report += "\n共 306 张 384×384 PNG。采样步骤为 12、24……408，对应 34 个时刻，相邻采样间隔 0.2 秒；索引从零开始，因此第 12 步对应源数据索引 11、时间 0.183333 秒。\n\n"
    report += "[打开三视角浏览页](index.html) · [完整核验及图片 SHA256](visual_audit.json) · [重播报告](summary.json)\n\n"
    report += "GIF 和总览图由本次原生 PNG 派生，原图保持不变。GIF 每帧 200 ms，与源数据采样间隔一致；实际 GUI 重播的墙钟耗时受渲染影响。\n\n"
    report += "本次恢复记录中的机械臂、手指和物体状态，不推进物理、不重新规划。所有源文件 SHA256 前后一致。历史记录范围止于抬升后保持，未包含运输与放置。\n"
    report += "\n人工检查：两路外部相机可以看到机械臂和被抬升物体。已检查的手眼相机抓取/抬升帧朝向室外背景，原样保留，未用于动作核验。\n"
    (root / "REPLAY_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({k: v for k, v in audit.items() if k != "images"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
