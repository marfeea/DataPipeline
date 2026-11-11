"""Audited isolated Legacy IK timed-reference lift with visible native GUI."""
import argparse
from datetime import datetime,timezone,timedelta
import difflib
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
BASE=HERE.parent/"legacy_lift_speed_20261015"
sys.path[:0]=[str(HERE),str(BASE),str(PROJECT)]
spec=importlib.util.spec_from_file_location("timed_lift_base",BASE/"run_sweep.py")
base=importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
sys.path.insert(0,str(HERE))


def now(): return datetime.now(timezone(timedelta(hours=8))).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data): Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
def production_hashes():
    paths=["scripts/probe_curobo_v2_legacy_contact_isaac.py","scripts/native/aubo_ik.py","scripts/native/ik_controller.py",
           "scripts/sim_backend.py","scripts/native/scene.py","configs/aubo_ik_control.py","configs/scene.py","scripts/collect_isaacsim.py",
           "configs/curobo/assets/aubo_e5_project/aubo_e5_grasp_tcp_20261015.urdf"]
    return {name:sha(PROJECT/name) for name in paths}


def prepare():
    old_here=base.HERE
    base.HERE=HERE
    try: source,manifest=base.prepare()
    finally: base.HERE=old_here
    changes=[]
    def replace(old,new):
        nonlocal source
        if source.count(old)!=1: raise RuntimeError(f"Upstream shape changed: {old[:90]}")
        source=source.replace(old,new)
        changes.append(dict(old=old,new=new))
    replace("from policy import bounded_lift_delta","from policy import bounded_lift_delta\nfrom timed_executor import run_lift")
    replace('parser.add_argument("--experiment-grasp-settle-s", type=float, default=0.0)',
            'parser.add_argument("--experiment-grasp-settle-s", type=float, default=0.0)\n'
            'parser.add_argument("--timed-lift-speed", type=float, default=0.25)\n'
            'parser.add_argument("--timed-lift-ramp", type=float, default=0.25)')
    replace('if record["stage"] not in ("free_space_to_pre_pick", "free_space_transfer"):',
            'if record["stage"] not in ("free_space_to_pre_pick", "free_space_transfer") and not record.get("timed_lift"):')
    replace('        report["lift"] = approach("legacy_lift", lift, -1.0, target_quaternion_b=pick_quaternion)\n'
            '        for _ in range(round(task["control_hz"] * 0.5)):\n'
            '            step_ik(lift, "legacy_lift_hold", -1.0, target_quaternion_b=pick_quaternion)',
            '        report["lift"] = run_lift(env=env,robot=robot,target=target,ik=ik,gripper=gripper,\n'
            '            arm_ids=arm_ids,cfg=cfg,task=task,args=args,records=records,report=report,\n'
            '            lift=lift,pick_quaternion=pick_quaternion,capture_sample=capture_sample,contacts=contact_reports)')
    source,n=re.subn(r'        report\["lift_experiment"\] = dict\(.*?source_probe_sha256="[^"\n]+"\)\n',"",source,flags=re.S)
    if n!=1: raise RuntimeError("Unable to remove unused increment-mode report")
    target=HERE/"generated_timed_probe.py"
    compile(source,str(target),"exec")
    target.write_text(source,encoding="utf-8")
    original=Path(manifest["upstream"]).read_text(encoding="utf-8")
    patch="".join(difflib.unified_diff(original.splitlines(True),source.splitlines(True),
                                    fromfile="production/probe_curobo_v2_legacy_contact_isaac.py",
                                    tofile="experiment/generated_timed_probe.py"))
    (HERE/"source.patch").write_text(patch,encoding="utf-8")
    manifest.update(generated_timed_sha256=hashlib.sha256(source.encode()).hexdigest(),timed_replacements=changes)
    write(HERE/"source_manifest.json",manifest)
    return source,manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id",default="legacy_timed_lift_20261021")
    parser.add_argument("--trials",type=int,default=3)
    parser.add_argument("--speed",type=float,default=.25)
    parser.add_argument("--ramp",type=float,default=.25)
    parser.add_argument("--prepare-only",action="store_true")
    args=parser.parse_args()
    if not 1<=args.trials<=5 or not 0<args.speed<=.6 or not .1<=args.ramp<=2:
        parser.error("Invalid trial count, speed, or ramp")
    source,manifest=prepare()
    if args.prepare_only:
        print(json.dumps(manifest,indent=2));return
    evroot=PROJECT/"data/test/trajdata"/args.run_id/"0_to_1"
    trroot=PROJECT/"data/test/trajectorys"/args.run_id/"0_to_1"
    evroot.mkdir(parents=True,exist_ok=False)
    trroot.mkdir(parents=True,exist_ok=False)
    def event(kind,**data):
        with (evroot/"events.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(dict(time=now(),event=kind,**data),ensure_ascii=False)+"\n")
    before=production_hashes()
    write(evroot/"production_before.json",before)
    snapshot=evroot/"experiment_source"
    snapshot.mkdir()
    files=[p for p in HERE.iterdir() if p.is_file() and p.suffix in (".py",".md",".json",".patch")]
    for path in files: shutil.copy2(path,snapshot/path.name)
    shutil.copy2(BASE/"run_sweep.py",snapshot/"base_run_sweep.py")
    shutil.copy2(BASE/"policy.py",snapshot/"policy.py")
    write(evroot/"source_inventory.json",{p.name:sha(p) for p in snapshot.iterdir()})
    write(evroot/"environment.json",dict(python=sys.executable,version=sys.version,argv=sys.argv,created_at=now()))
    runtime=Path(sys.executable).parent
    os.environ["PATH"]=os.pathsep.join([str(runtime),str(runtime/"Scripts"),str(runtime/"Library/bin"),os.environ.get("PATH","")])
    reference=PROJECT/"data/0_to_1_1014_tcp/0_to_1_evidence"
    inputs=[reference/"first_probe_01/report.json",reference/"targets.json",reference/"world/report.json"]
    collision_model=Path(base.read(inputs[2])["collision_model"])
    write(evroot/"input_inventory.json",{str(p):sha(p) for p in [*inputs,collision_model]})
    status=dict(mode="gui",scope="Legacy IK timed lift; no main integration",trials=[],settings=vars(args),manifest=manifest)
    write(evroot/"status.json",status)
    event("run_started",settings=vars(args))
    cache=base.install_shared_runtime(evroot,camera_warmup_timeout_s=60)
    try:
        for index in range(args.trials):
            label=f"trial_{index+1:02d}"
            ev,tr=evroot/label,trroot/label
            command=[str(HERE/"generated_timed_probe.py"),"--backend","isaacsim","--device","cuda:0",
                     "--first-probe",str(inputs[0]),"--targets-report",str(inputs[1]),"--world-report",str(inputs[2]),
                     "--output",str(ev),"--trajectory-output",str(tr),"--stop-after-lift","--capture-execution",
                     "--camera-width","256","--camera-height","256","--physics-hz","120",
                     "--contact-velocity-iterations","4","--contact-policy","run003",
                     "--pick-lift-step-m","0.001","--pick-lift-command-hz","60","--pick-lift-speed-mode","step",
                     "--experiment-grasp-settle-s","0.35","--experiment-lift-step-m","0.01",
                     "--experiment-lift-command-hz","60","--timed-lift-speed",str(args.speed),"--timed-lift-ramp",str(args.ramp)]
            entry=dict(label=label,command=[sys.executable,*command],evidence_directory=str(ev),trajectory_directory=str(tr))
            status["trials"].append(entry)
            write(evroot/"status.json",status)
            event("trial_started",label=label,command=entry["command"])
            print("TIMED_TRIAL_START "+label,flush=True)
            sys.argv=command
            try: exec(compile(source,command[0],"exec"),{"__name__":"__main__","__file__":command[0]})
            except SystemExit as result: entry["exit_code"]=result.code
            if (ev/"report.json").exists():
                report=base.read(ev/"report.json")
                entry.update(execution_passed=report["passed"],failure_reason=report.get("failure_reason"),record_count=report.get("record_count",0))
            write(evroot/"status.json",status)
            event("trial_finished",**entry)
            print("TIMED_TRIAL_FINISH "+json.dumps(entry),flush=True)
            if not entry.get("record_count"): break
            gc.collect()
            import torch
            torch.cuda.empty_cache()
    finally:
        after=production_hashes()
        write(evroot/"production_after.json",after)
        write(evroot/"production_unchanged.json",dict(unchanged=before==after,changed=[k for k in before if before[k]!=after[k]]))
        event("run_finished",production_unchanged=before==after)
        if "real_close" in cache: cache["real_close"]()
        if "launcher" in cache: cache["launcher"].app.close()


if __name__=="__main__": main()
