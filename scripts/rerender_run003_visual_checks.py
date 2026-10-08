"""After graph jobs exit, replay existing trajectories into separate evidence folders."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--routes", nargs="+", required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    status = {"stage": "waiting", "wait_pid": args.wait_pid, "routes": {}}
    status_path = output / "supplemental_visual_replay_status.json"

    def save():
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel.WaitForSingleObject.restype = ctypes.c_ulong
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x00100000, False, args.wait_pid)
    if not handle:
        raise RuntimeError("Cannot verify graph process handle; supplemental replays not started")
    save()
    print(f"WAIT graph helper pid={args.wait_pid}", flush=True)
    try:
        while True:
            code = kernel.WaitForSingleObject(handle, 50000)
            if code == 0:
                break
            if code != 258:
                raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle(handle)
    runtime = Path(sys.executable).parent
    os.environ["PATH"] = os.pathsep.join([str(runtime), str(runtime / "Scripts"), str(runtime / "Library/bin"), os.environ.get("PATH", "")])
    scripts = Path(__file__).resolve().parent
    for name in args.routes:
        if name not in {f"{i}_to_{j}" for i in range(5) for j in range(5) if i != j}:
            raise ValueError("invalid directed route")
        run = output / name
        report_name = "replay_visual_recheck_report.json"
        destination = run / "replay_visualization_recheck"
        if destination.exists():
            raise FileExistsError("Preserve prior supplemental evidence; select a new output directory")
        status["stage"] = name
        save()
        command = [sys.executable, str(scripts / "replay_curobo_pick_place_v2_isaac.py"),
                   "--run", str(run), "--no-realtime", "--device", "cuda:0", "--headless",
                   "--report-name", report_name, "--visualization-dir", str(destination)]
        print(f"VISUAL_REPLAY {name}", flush=True)
        with (output / f"{name}_visual_recheck.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=scripts.parent, stdout=log, stderr=subprocess.STDOUT)
        report = json.loads((run / report_name).read_text()) if (run / report_name).exists() else {}
        status["routes"][name] = {"exit_code": result.returncode, "state_replay_passed": report.get("passed", False),
                                  "report": str(run / report_name), "visualization": str(destination)}
        save()
        print(f"VISUAL_REPLAYED {name} exit={result.returncode}", flush=True)
    status["stage"] = "complete"
    status["scope"] = "same trajectory supplemental state replay; image content needs visual inspection"
    save()


if __name__ == "__main__":
    main()
