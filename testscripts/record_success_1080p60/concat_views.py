"""Join complete camera clips in front, side, wrist order and verify output."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import cv2
import imageio_ffmpeg

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    folder=args.directory.resolve()
    source=json.loads((folder/'video_report.json').read_text(encoding='utf-8'))
    assert source['passed']
    order=['CameraSensor_3','CameraSensor_2','HandCamera']
    clips=[folder/(name+'.mp4') for name in order]
    manifest=folder/'concat_inputs.txt'
    manifest.write_text(''.join("file '"+p.as_posix().replace("'","'\\''")+"'\n" for p in clips),encoding='utf-8')
    result=folder/'v1.3_trial_01_front_side_hand_1080p60.mp4'
    with (folder/'concat_encoder.log').open('wb') as log:
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','concat','-safe','0','-i',str(manifest),'-c','copy','-movflags','+faststart',str(result)],stdout=log,stderr=subprocess.STDOUT,check=True)
    cap=cv2.VideoCapture(str(result))
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    metadata=dict(path=str(result),width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),fps=cap.get(cv2.CAP_PROP_FPS),frames=count)
    n=source['source_frames']
    assert (metadata['width'],metadata['height'],metadata['fps'],count)==(1920,1080,60.,3*n),metadata
    for index in [0,n//2,n-1,n,n+n//2,2*n-1,2*n,2*n+n//2,3*n-1]:
        cap.set(cv2.CAP_PROP_POS_FRAMES,index)
        ok,pixels=cap.read()
        assert ok and pixels[::16,::16].std()>5,index
    cap.release()
    metadata.update(passed=True,duration_s=count/60,segments=[dict(camera=name,start_s=i*n/60,end_s=(i+1)*n/60,source_sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for i,(name,p) in enumerate(zip(order,clips))],source_report=str(folder/'video_report.json'))
    (folder/'concat_report.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(metadata,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
