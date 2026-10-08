import hashlib
import xml.etree.ElementTree as ET
from math import sqrt

import numpy as np
import pytest

from DataPipe.pipelines.curobo_bootstrap.robot_adapter import UrdfChain, pose_matrix
from DataPipe.pipelines.curobo_bootstrap.robot_asset import derive_robot_asset


@pytest.fixture
def source_and_report(tmp_path):
    root = ET.Element("robot", name="test")
    names = ["AUBO_E5", "l1", "l2", "l3", "l4", "l5", "AUBO_E5_Flange"]
    for name in names:
        ET.SubElement(root, "link", name=name)
    for i in range(6):
        joint = ET.SubElement(root, "joint", name=f"j{i}", type="revolute")
        ET.SubElement(joint, "parent", link=names[i])
        ET.SubElement(joint, "child", link=names[i + 1])
        ET.SubElement(joint, "limit", lower="-3", upper="3", velocity="2")
    source = tmp_path / "source.urdf"
    source.write_bytes(ET.tostring(root))
    report = {
        "urdf_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "isaac_contract": {"joint_names": [f"j{i}" for i in range(6)], "velocity": [1, 3, 1, 3, 1, 3]},
    }
    return source, report


def derive(source, destination, report):
    return derive_robot_asset(
        source,
        destination,
        report,
        tcp_position=(-0.17086, 0, 0.11305),
        tcp_quaternion_wxyz=(sqrt(0.5), 0, sqrt(0.5), 0),
    )


def test_derivation_preserves_source_and_restricts_velocity(source_and_report, tmp_path):
    source, report = source_and_report
    original = source.read_bytes()
    output = tmp_path / "derived"
    manifest = derive(source, output, report)
    assert source.read_bytes() == original
    assert manifest["source_sha256"] == report["urdf_sha256"]
    assert [r["effective_velocity"] for r in manifest["velocity_changes"]] == [1, 2, 1, 2, 1, 2]
    actual = UrdfChain(output / "aubo_e5_project.urdf", "AUBO_E5_Flange", "project_tcp").fk([])
    np.testing.assert_allclose(actual, pose_matrix((-0.17086, 0, 0.11305), (sqrt(0.5), 0, sqrt(0.5), 0)), atol=1e-15)
    with pytest.raises(FileExistsError):
        derive(source, output, report)


@pytest.mark.parametrize("corruption", ["hash", "order", "velocity"])
def test_rejects_stale_or_invalid_evidence(source_and_report, tmp_path, corruption):
    source, report = source_and_report
    if corruption == "hash":
        report["urdf_sha256"] = "bad"
    elif corruption == "order":
        report["isaac_contract"]["joint_names"].reverse()
    else:
        report["isaac_contract"]["velocity"][0] = float("nan")
    output = tmp_path / "derived"
    with pytest.raises(ValueError):
        derive(source, output, report)
    assert not output.exists()
