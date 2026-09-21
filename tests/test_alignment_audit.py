import runpy
from pathlib import Path

import h5py
from test_alignment_v4 import aligned_episode, writer

audit_module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/audit_alignment_dataset.py"))


def test_acceptance_inventory_includes_hard_link_aliases(tmp_path):
    writer(tmp_path).write(aligned_episode())
    with h5py.File(tmp_path / "shard-00000.hdf5") as handle:
        fields = audit_module["field_shapes"](handle["data/demo_aligned"])
        assert fields["actions"] == fields["action_commanded"] == fields["diagnostics/action_commanded"] == ["T", 8]
        assert fields["transforms/T_w_b"] == [4, 4]
    report = audit_module["audit"](tmp_path)
    assert not report["passed"]
    assert any("missing file metadata task_id" in error for error in report["errors"])


def test_acceptance_reports_missing_dataset(tmp_path):
    report = audit_module["audit"](tmp_path)
    assert not report["passed"]
    assert report["errors"] == ["missing manifest.json"]
