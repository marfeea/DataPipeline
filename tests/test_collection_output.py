import pytest
from test_alignment_v4 import aligned_episode, writer
from test_datapipe_p0_r import _episode

from DataPipe.collection_output import resolve_collection_output
from DataPipe.config import ALIGNED_SCHEMA_VERSION, DatasetConfig
from DataPipe.hdf5_writer import HDF5TrajectoryWriter
from DataPipe.validator import validate_hdf5_dataset


def test_legacy_redirect_preserves_data_and_resumes_v4(tmp_path):
    legacy = tmp_path / "pick_place_smoke_4_to_0"
    HDF5TrajectoryWriter(legacy, DatasetConfig(min_free_disk_bytes=0)).write(_episode("old", steps=2))
    before = {p.name: p.read_bytes() for p in legacy.iterdir()}
    selected = resolve_collection_output(legacy)
    assert selected == tmp_path / "pick_place_smoke_4_to_0_v4"
    assert not selected.exists()
    writer(selected).write(aligned_episode("new"))
    assert resolve_collection_output(legacy) == selected
    resumed = writer(resolve_collection_output(legacy))
    assert resumed.episode_ids == frozenset({"new"})
    resumed.write(aligned_episode("next"))
    assert validate_hdf5_dataset(selected).passed
    assert before == {p.name: p.read_bytes() for p in legacy.iterdir()}
    with pytest.raises(ValueError, match="incompatible dataset schema"):
        HDF5TrajectoryWriter(legacy, DatasetConfig(schema_version=ALIGNED_SCHEMA_VERSION))


def test_fresh_and_matching_output_unchanged(tmp_path):
    fresh = tmp_path / "fresh"
    assert resolve_collection_output(fresh) == fresh
    writer(fresh).write(aligned_episode())
    assert resolve_collection_output(fresh) == fresh


@pytest.mark.parametrize("content", ["{bad json", "{}", '{"schema_version": "unknown"}'])
def test_unreadable_or_unknown_manifest_is_not_bypassed(tmp_path, content):
    (tmp_path / "manifest.json").write_text(content)
    with pytest.raises(ValueError):
        resolve_collection_output(tmp_path)


def test_conflicting_sibling_is_not_overwritten(tmp_path):
    legacy = tmp_path / "old"
    sibling = tmp_path / "old_v4"
    for path in (legacy, sibling):
        HDF5TrajectoryWriter(path, DatasetConfig(min_free_disk_bytes=0)).write(_episode("old", steps=2))
    with pytest.raises(ValueError, match="already contains schema"):
        resolve_collection_output(legacy)


def test_orphan_shards_are_not_overwritten(tmp_path):
    (tmp_path / "shard-00000.hdf5").write_bytes(b"preserve")
    with pytest.raises(ValueError, match="without manifest"):
        resolve_collection_output(tmp_path)
    assert (tmp_path / "shard-00000.hdf5").read_bytes() == b"preserve"
