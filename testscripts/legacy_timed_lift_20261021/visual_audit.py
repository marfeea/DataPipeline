"""Index representative original renders for every successful or failed lift trial."""
import hashlib
import json
from pathlib import Path
from PIL import Image,ImageDraw

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
IDS=["legacy_timed_lift_20261021_pilot","legacy_timed_lift_20261021_validation","legacy_timed_lift_20261021_smooth","legacy_timed_lift_20261021_verified"]
ROOT=PROJECT/"data/test/trajdata"/IDS[-1]/"0_to_1"

def main():
    chosen=[]
    for run_id in IDS:
        root=PROJECT/"data/test/trajdata"/run_id/"0_to_1"
        summary=json.loads((root/"summary.json").read_text(encoding="utf-8"))
        sheet=Image.new("RGB",(768,len(summary["results"])*284),"white")
        draw=ImageDraw.Draw(sheet)
        for row,result in enumerate(summary["results"]):
            manifest=Path(result["evidence_directory"])/"execution_visualization/frames.json"
            data=json.loads(manifest.read_text(encoding="utf-8"))
            for column,(stage,camera) in enumerate([("close_gripper","CameraSensor_3"),("legacy_lift","CameraSensor_2"),("legacy_lift_hold","CameraSensor_3")]):
                frames=[f for f in data["frames"] if f["execution_stage"]==stage and f["camera"]==camera]
                frame=frames[len(frames)//2] if stage=="legacy_lift" else frames[-1]
                path=manifest.parent/frame["rgb"]
                with Image.open(path) as image:
                    sheet.paste(image.convert("RGB").resize((256,256)),(column*256,row*284))
                draw.text((column*256+3,row*284+257),f'{result["label"]} | {stage.replace("legacy_","")} #{frame["trajectory_index"]}',fill="black")
                chosen.append(dict(run_id=run_id,trial=result["label"],stage=stage,camera=camera,
                                   trajectory_index=frame["trajectory_index"],time_s=frame["time_s"],image=str(path),
                                   sensor_render_reference=frame["sensor_render_reference"],sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        sheet.save(ROOT/f"native_overview_{run_id}.png")
    (ROOT/"visual_audit_index.json").write_text(json.dumps(chosen,indent=2),encoding="utf-8")
    print(f"Indexed {len(chosen)} representative native images from 10 actual lift trials.")

if __name__=="__main__":main()
