"""Collect the remaining incoming state4 routes with frozen sources and GUI evidence."""

import argparse
import json
from pathlib import Path

from collect_state4_outgoing_routes import read, run_script, verify_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    args = parser.parse_args()
    batch = args.batch.resolve()
    manifest = read(batch / "experiment_manifest.json")
    assert manifest["schema"] == "state4_incoming_collection/1"
    routes = manifest["routes"]
    assert routes and len(set(routes)) == len(routes)
    assert set(routes) <= {"0_to_4", "1_to_4", "2_to_4", "3_to_4"}
    assert set(manifest["route_parameters"]) == set(routes)
    verify_sources(manifest)
    records = []
    result_path = batch / "collection_result.json"

    def save():
        result_path.write_text(
            json.dumps(
                dict(
                    schema="state4_incoming_collection_result/1",
                    derived_from_run=manifest["derived_from_run"],
                    records=records,
                    frozen_sources_verified=True,
                    complete=len(records) == len(routes) and all(record["passed"] for record in records),
                    scope="run003 mixed planning/contact execution and GUI state replay; trays remain diagnostic",
                ),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    for route in manifest["routes"]:
        verify_sources(manifest)
        run = batch / route
        evidence = batch / (route + "_evidence")
        parameters = manifest["route_parameters"][route]
        record = dict(route=route, passed=False, stage="collection", run=str(run))
        records.append(record)
        save()
        try:
            route_report = evidence / "run_report.json"
            if not route_report.exists():
                arguments = []
                for key, value in parameters.items():
                    if isinstance(value, bool):
                        if value:
                            arguments.append("--" + key.replace("_", "-"))
                    else:
                        arguments += ["--" + key.replace("_", "-"), str(value)]
                arguments += ["--output", str(run)]
                run_script("collect_isaacsim.py", arguments, batch / (route + "_collection.log"))
            report = read(route_report)
            if not report["passed"]:
                raise ValueError(f"collection failed at {report['stage']}: {report.get('failure_reason')}")
            record["stage"] = "replay"
            save()
            if not (run / "replay_report.json").exists():
                run_script(
                    "replay_isaacsim.py",
                    ["--run", run, "--device", "cuda:0", "--camera-width", "640", "--camera-height", "480"],
                    run / "replay_gui.log",
                )
            assert read(run / "replay_report.json")["passed"]
            record["stage"] = "audit"
            save()
            captures = []
            for frames_path in sorted(evidence.glob("*/execution_visualization/frames.json")):
                if not read(frames_path)["frames"]:
                    continue
                probe = frames_path.parents[1]
                run_script("audit_contact_execution_visuals.py", ["--run", probe], probe / "capture_audit.log")
                assert read(probe / "capture_audit.json")["capture_audit_passed"]
                captures.append(dict(run=str(probe), images=len(read(frames_path)["frames"])))
            run_script("preview_run003_stage1_gui.py", ["--run", run], run / "preview_audit.log")
            run_script("audit_state4_routed_run.py", ["--run", run], run / "state4_routing_audit.log")
            audit = read(run / "state4_routing_audit.json")
            assert audit["audit_passed"]
            integrated = Path(report["integrated_run"])
            record.update(
                passed=True,
                stage="complete",
                samples=report["samples"],
                task_hash=report["task_hash"],
                accepted_candidate=integrated.name.removeprefix("integrated_"),
                final_target_position_error_m=audit["final_target_error_m"],
                native_images=len(read(integrated / "execution_visualization/frames.json")["frames"]),
                replay_images=audit["replay_images"],
                audited_captures=captures,
                state4_routing=audit["state4_routing"],
                source2_routing=read(evidence / "targets.json")["task_snapshot"].get("source2_routing"),
            )
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
        verify_sources(manifest)
        save()
        print("STATE4_INCOMING_RESULT " + json.dumps(record), flush=True)
    return 0 if all(record["passed"] for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
