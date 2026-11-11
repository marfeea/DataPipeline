"""Comparison of measured speed, reference lead, and grasp drift in simulation time."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
sys.path.insert(0,str(PROJECT))
from scripts.pipelines.curobo_bootstrap.robot_adapter import pose_matrix

def pose(p,q):
    q=np.asarray(q,float)
    return pose_matrix(p,q/np.linalg.norm(q))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id",default="legacy_timed_lift_20261021_smooth")
    args=parser.parse_args()
    root=PROJECT/"data/test/trajdata"/args.run_id/"0_to_1"
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    ids=[("legacy_timed_lift_20261021_pilot","0.25m/s ramp0.25"),
         ("legacy_timed_lift_20261021_validation","0.35m/s ramp0.45"),
         (args.run_id,"0.40m/s ramp0.80")]
    for run_id,label in ids:
        ev=PROJECT/"data/test/trajdata"/run_id/"0_to_1/trial_01"
        if not (ev/"report.json").exists():continue
        report=json.loads((ev/"report.json").read_text())
        trace=json.loads(Path(report["trace_path"]).read_text())
        ii=[i for i,r in enumerate(trace) if r.get("timed_lift")]
        if not ii:continue
        rows=[trace[ii[0]-1],*[trace[i] for i in ii]]
        base=pose(report["base_pose_w"][:3],report["base_pose_w"][3:])
        p=np.array([r["tcp_position_b"] for r in rows])
        pw=p@base[:3,:3].T+base[:3,3]
        times=np.arange(len(rows))/60
        axes[0,0].plot(times,pw[:,2]-pw[0,2],label=label)
        axes[0,1].plot(times[1:],np.linalg.norm(np.diff(pw,axis=0),axis=1)*60*100,label=label)
        leads=[np.linalg.norm(np.array(r["reference_tcp_position_b"])-r["current_tcp_position_b"])*100 for r in rows[1:]]
        axes[1,0].plot(times[1:],leads,label=label)
        relative=[]
        for r in rows:
            tcp=base@pose(r["tcp_position_b"],r["tcp_quaternion_b"])
            obj=pose(r["target_pose_w"][:3],r["target_pose_w"][3:])
            relative.append(np.linalg.inv(tcp)@obj)
        drift=[np.linalg.norm(r[:3,3]-relative[0][:3,3])*1000 for r in relative]
        axes[1,1].plot(times,drift,label=label)
    axes[1,0].axhline(4,color="black",linestyle=":",label="slow clock at 4cm")
    axes[1,0].axhline(6,color="red",linestyle=":",label="lead cap 6cm")
    axes[1,1].axhline(5,color="red",linestyle=":",label="acceptance 5mm")
    for ax,(title,y) in zip(axes.flat,[("Measured lift height","Height (m)"),("Measured TCP speed","Speed (cm/s)"),
                                      ("Time reference ahead of actual TCP","Tracking lead (cm)"),("Object relative to TCP","Translation drift (mm)")]):
        ax.set(title=title,xlabel="Physics simulation time (s)",ylabel=y)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.savefig(root/"comparison.png",dpi=160)
    print(str(root/"comparison.png"))

if __name__=="__main__":main()
