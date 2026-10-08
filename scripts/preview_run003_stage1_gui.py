"""Make an inspection sheet from unchanged GUI replay frames at four sample times."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    report = json.loads((run / "replay_report.json").read_text(encoding="utf-8"))
    assert report["passed"] and report["mode"] == "gui"
    visual = run / "replay_visualization"
    frames = json.loads((visual / "frames.json").read_text(encoding="utf-8"))["frames"]
    cameras = ("HandCamera", "CameraSensor_2", "CameraSensor_3")
    indices = sorted({frame["trajectory_index"] for frame in frames})
    chosen = [indices[round(fraction * (len(indices) - 1))] for fraction in (0, 0.3, 0.65, 1)]
    sheet = Image.new("RGB", (996, 1128), "#eeeeee")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
    draw.text((12, 8), f"{run.name} | GUI | source RGB 640x480 | sampled frames", font=font, fill="black")
    selected = []
    for row, index in enumerate(chosen):
        y = 40 + row * 272
        draw.text((12, y), f"trajectory index {index}", font=font, fill="black")
        for col, camera in enumerate(cameras):
            frame = next(frame for frame in frames if frame["trajectory_index"] == index and frame["camera"] == camera)
            with Image.open(visual / frame["rgb"]) as image:
                assert image.size == (640, 480)
                sheet.paste(image.convert("RGB").resize((320, 240), Image.Resampling.LANCZOS), (12 + col * 328, y + 26))
            draw.text((16 + col * 328, y + 28), camera, font=font, fill="white", stroke_width=1, stroke_fill="black")
            selected.append(dict(camera=camera, trajectory_index=index, rgb=str(visual / frame["rgb"])))
    output = run.parent / (run.name + "_gui_preview.jpg")
    sheet.save(output, quality=94)
    (run.parent / (run.name + "_visual_inspection_samples.json")).write_text(
        json.dumps(
            dict(scope="four sampled times per camera, not all-frame content acceptance", frames=selected), indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
