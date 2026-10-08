"""Wait for the live first-pass collector, retry runtime failures, and summarize."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_root.resolve()
    status_path = output / "retry_pipeline_status.json"
    status = {"wait_pid": args.wait_pid, "stage": "opening_process", "passed": False}

    def save() -> None:
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel.WaitForSingleObject.restype = ctypes.c_ulong
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x00100000, False, args.wait_pid)
    if not handle:
        status["failure_reason"] = str(ctypes.WinError(ctypes.get_last_error()))
        save()
        raise RuntimeError("Cannot verify the first-pass process handle; retry was not started")
    status["stage"] = "waiting_on_live_process_handle"
    save()
    print(f"WAIT first-pass pid={args.wait_pid}", flush=True)
    try:
        while True:
            result = kernel.WaitForSingleObject(handle, 50000)
            if result == 0:
                break
            if result != 258:
                raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle(handle)

    runtime = Path(sys.executable).parent
    os.environ["PATH"] = os.pathsep.join([
        str(runtime), str(runtime / "Scripts"), str(runtime / "Library/bin"),
        os.environ.get("PATH", ""),
    ])
    status["stage"] = "retrying_environment_failures"
    save()
    command = [sys.executable, str(root / "scripts/collect_curobo_pick_place_v2_graph.py"),
               "--output-root", str(output), "--contact-policy", "run003",
               "--collision-model", str(output / "collision_model.json"),
               "--seed-count", "200", "--max-candidates", "8",
               "--retry-environment-failures"]
    with (output / "environment_retry.log").open("w", encoding="utf-8") as stream:
        retry = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
    status["retry_exit_code"] = retry.returncode
    status["stage"] = "summarizing"
    save()
    summary = subprocess.run([sys.executable, str(root / "scripts/summarize_curobo_pick_place_v2_graph.py"),
                              "--output-root", str(output)], cwd=root)
    status["summary_exit_code"] = summary.returncode
    status["stage"] = "complete"
    status["passed"] = retry.returncode == summary.returncode == 0
    status["scope"] = "collector and summary finished; full graph completion still requires independent evidence audit"
    save()
    return 0 if status["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
