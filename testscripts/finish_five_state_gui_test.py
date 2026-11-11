"""Wait for GUI recovery, validate one source2 MVP, then expand and independently audit."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from testscripts.script_paths import script_path

import argparse
import ctypes
import json
import os
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--wait-pid", type=int, help="Optional running GUI recovery process to wait on")
    args = parser.parse_args()
    batch = args.output_root.resolve()
    status_path = batch / "source2_recovery_status.json"
    status = dict(schema="source2_gui_recovery_status/1", phase="waiting_for_gui_recovery", wait_pid=args.wait_pid, passed=False)
    environment = dict(os.environ, HEADLESS="0", PYTHONPATH=str(ROOT))

    def save():
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    def run(script, arguments, log_name):
        command = [sys.executable, str(script_path(script)), *map(str, arguments)]
        assert "--headless" not in command and "--no-gui" not in command
        print("COMMAND " + subprocess.list2cmdline(command), flush=True)
        with (batch / log_name).open("w", encoding="utf-8") as stream:
            code = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT).returncode
        status.setdefault("commands", []).append(dict(command=command, returncode=code, log=str(batch / log_name)))
        save()
        return code

    def audit(routes, log_name):
        run("audit_five_state_gui_test.py", ["--output-root", batch, "--routes", *routes], log_name)
        results = read(batch / "independent_audit.json")["routes"]
        failed = {name: results[name] for name in routes if not results[name]["passed"]}
        if failed:
            raise RuntimeError("source2 route audit failed: " + json.dumps(failed))

    save()
    try:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        kernel.WaitForSingleObject.restype = ctypes.c_ulong
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x00100000, False, args.wait_pid) if args.wait_pid is not None else None
        if args.wait_pid is not None and not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            while handle:
                result = kernel.WaitForSingleObject(handle, 10000)
                if result == 0:
                    break
                if result != 258:
                    raise ctypes.WinError(ctypes.get_last_error())
        finally:
            if handle:
                kernel.CloseHandle(handle)
        recovered = read(batch / "independent_audit.json")["routes"]
        assert all(recovered[name]["passed"] for name in ("4_to_0", "4_to_2", "3_to_1"))
        status["phase"] = "source2_mvp_2_to_0"
        save()
        run("collect_five_state_gui_test.py", ["--output-root", batch, "--routes", "2_to_0", "--source2-recovery"], "source2_mvp_collection.log")
        audit(["2_to_0"], "source2_mvp_audit.log")
        status["initial_route_passed"] = True
        status["phase"] = "source2_expansion"
        save()
        run("collect_five_state_gui_test.py", ["--output-root", batch, "--routes", "2_to_1", "2_to_3", "2_to_4", "--source2-recovery"], "source2_expansion_collection.log")
        audit(["2_to_1", "2_to_3", "2_to_4"], "source2_expansion_audit.log")
        status["phase"] = "final_full_audit"
        save()
        run("audit_five_state_gui_test.py", ["--output-root", batch], "final_independent_audit.log")
        status["passed"] = read(batch / "independent_audit.json")["passed"]
        status["phase"] = "complete"
        save()
        return 0 if status["passed"] else 1
    except Exception as error:
        status.update(failed_phase=status["phase"], phase="failed", failure_reason=f"{type(error).__name__}: {error}")
        save()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
