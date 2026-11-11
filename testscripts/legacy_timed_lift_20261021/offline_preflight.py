"""Numerical preflight on a prior measured start; not a physical execution result."""
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(PROJECT)]
from timed_reference import Settings,plan_reference

def main():
    prior=PROJECT/"data/test/trajdata/curobo_lift_20261015_line_scaled/0_to_1/line_time_1p25/report.json"
    meta=json.loads(prior.read_text(encoding="utf-8"))["curobo_lift_experiment"]
    result=[]
    for speed,ramp in ((.25,.25),(.30,.35),(.35,.45),(.40,.5),(.35,.6),(.40,.65),(.40,.8),(.45,.8)):
        ref=plan_reference(meta["start_q"],meta["requested_goal_tcp_pose_b"][:3],meta["requested_goal_tcp_pose_b"][3:],Settings(speed_m_s=speed,ramp_s=ramp))
        result.append(ref.metadata)
    (HERE/"offline_preflight.json").write_text(json.dumps(dict(scope="numerical path preflight only; not new simulation evidence",source=str(prior),results=result),indent=2),encoding="utf-8")
    print(json.dumps([dict(speed=r["settings"]["speed_m_s"],ramp=r["settings"]["ramp_s"],scale=r["time_scale"],duration=r["duration_s"]) for r in result],indent=2))

if __name__=="__main__": main()
