"""Independent post-process audit after native SimulationApp.close exits Python."""
from datetime import datetime,timezone,timedelta
import difflib
import hashlib
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
IDS=["legacy_timed_lift_20261021_pilot","legacy_timed_lift_20261021_validation", "legacy_timed_lift_20261021_smooth","legacy_timed_lift_20261021_verified"]

def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding="utf-8")

def main():
    results=[]
    for run_id in IDS:
        root=PROJECT/"data/test/trajdata"/run_id/"0_to_1"
        before=read(root/"production_before.json")
        after={p:sha(PROJECT/p) for p in before}
        inventory=read(root/"source_inventory.json")
        inputs=read(root/"input_inventory.json")
        now=datetime.now(timezone(timedelta(hours=8))).isoformat()
        audit=dict(unchanged=before==after,changed=[k for k in before if before[k]!=after[k]],
                   source_snapshot_verified=all(sha(root/"experiment_source"/p)==h for p,h in inventory.items()),
                   input_files_unchanged=all(sha(p)==h for p,h in inputs.items()),
                   method="independent rehash after GUI process exit",verified_at=now,verifier_sha256=sha(__file__))
        write(root/"production_after.json",after)
        write(root/"production_unchanged.json",audit)
        folder=root/"audit_source"
        folder.mkdir(exist_ok=True)
        shutil.copy2(__file__,folder/Path(__file__).name)
        old=(root/"experiment_source/run_experiment.py").read_text(encoding="utf-8")
        new=(HERE/"run_experiment.py").read_text(encoding="utf-8")
        (folder/"audit_cleanup.patch").write_text("".join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
                                                             fromfile="executed/run_experiment.py",tofile="current/run_experiment.py")),encoding="utf-8")
        with (root/"events.jsonl").open("a",encoding="utf-8") as stream:
            stream.write(json.dumps(dict(time=now,event="postprocess_audit_completed",**audit),ensure_ascii=False)+"\n")
        results.append(dict(run_id=run_id,**audit))
        if not all(audit[k] for k in ("unchanged","source_snapshot_verified","input_files_unchanged")):
            raise RuntimeError(f"Audit mismatch in {run_id}")
    print(json.dumps(results,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
