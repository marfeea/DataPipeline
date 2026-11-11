"""Plot planned versus physical lift, keeping native images as motion evidence."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path[:0] = [str(HERE),str(PROJECT)]
from scripts.native.aubo_ik import forward_kinematics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id",default="curobo_lift_20261015_retry01")
    args = parser.parse_args()
    root = PROJECT/"data/test/trajdata"/args.run_id/"0_to_1"
    summary = json.loads((root/"summary.json").read_text())
    fig, axes = plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    colors = ["tab:blue","tab:orange","tab:purple","tab:green"]
    for index, result in enumerate(summary["results"]):
        if "lift_duration_s" not in result:
            continue
        color, label = colors[index], result["label"]
        trace = json.loads((Path(result["trajectory_directory"])/"trace.json").read_text())
        ii = [i for i,r in enumerate(trace) if r["stage"]=="legacy_lift"]
        rows = [trace[ii[0]-1],*[trace[i] for i in ii]]
        p = np.array([r["tcp_position_b"] for r in rows])
        t = np.arange(len(p))/60
        axes[0,0].plot(t,p[:,2]-p[0,2],label=label,color=color)
        axes[0,1].plot(t[1:],np.linalg.norm(np.diff(p,axis=0),axis=1)*60*100,color=color,label=label)
        ref = np.array([r["reference_tcp_position_b"] for r in rows[1:]])
        before = np.array([r["current_tcp_position_b"] for r in rows[1:]])
        axes[1,0].plot(t[1:],np.linalg.norm(ref-before,axis=1)*100,color=color,label=label)
        axes[1,1].plot((p[:,0]-p[0,0])*100,p[:,2]-p[0,2],label=label,color=color)
        with np.load(Path(result["trajectory_directory"])/"execution_reference.npz") as data:
            planned = np.array([forward_kinematics(q)[:3,3] for q in data["joint_position"]])
            pt = data["time_s"]
        axes[0,0].plot(pt,planned[:,2]-p[0,2],"--",color=color,alpha=.65,label=f"{label} planned")
        axes[0,1].plot(pt[1:],np.linalg.norm(np.diff(planned,axis=0),axis=1)*60*100,"--",color=color,alpha=.65)
    old = PROJECT/"data/test/trajectorys/legacy_lift_1cm_60hz_20261015/0_to_1/joint_0p785_interp_0p5000/trace.json"
    trace = json.loads(old.read_text())
    ii = [i for i,r in enumerate(trace) if r["stage"]=="legacy_lift"]
    rows = [trace[ii[0]-1],*[trace[i] for i in ii]]
    p = np.array([r["tcp_position_b"] for r in rows])
    t = np.arange(len(p))/60
    axes[0,0].plot(t,p[:,2]-p[0,2],color="black",label="Legacy IK 1cm 60Hz")
    axes[0,1].plot(t[1:],np.linalg.norm(np.diff(p,axis=0),axis=1)*60*100,color="black",label="Legacy IK 1cm 60Hz")
    axes[1,0].axhline(1,color="black",linestyle=":",label="Legacy relative target cap (1cm)")
    axes[1,1].plot((p[:,0]-p[0,0])*100,p[:,2]-p[0,2],color="black",label="Legacy IK 1cm 60Hz")
    labels = [("Time (s)","TCP lift height (m)","Planned / measured lift"),
              ("Time (s)","TCP path speed (cm/s)","Planned / measured speed, 60Hz samples"),
              ("Time (s)","Reference ahead of actual TCP (cm)","Time-indexed reference can build tracking lead"),
              ("TCP X displacement (cm)","TCP lift height (m)","Side-view path (not all 3D deviation)")]
    for ax, (x,y,title) in zip(axes.flat,labels):
        ax.set(xlabel=x,ylabel=y,title=title)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.savefig(root/"planned_vs_measured.png",dpi=170)
    with (root/"report.md").open("a",encoding="utf-8") as stream:
        stream.write(f'\n![规划、实测速度与跟踪超前量对比]({(root/"planned_vs_measured.png").as_posix()})\n')
    print(str(root/"planned_vs_measured.png"))


if __name__=="__main__":
    main()
