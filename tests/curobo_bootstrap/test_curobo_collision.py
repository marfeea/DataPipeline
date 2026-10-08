import numpy as np
import pytest

from DataPipe.pipelines.curobo_bootstrap.curobo_collision import pack_attached_spheres
from DataPipe.pipelines.curobo_bootstrap.world_adapter import AttachedSphere


def test_pack_attached_spheres_uses_inactive_negative_radius_slots():
    packed = pack_attached_spheres((AttachedSphere((1, 2, 3), 0.1),), 3)
    np.testing.assert_allclose(packed[0], [1, 2, 3, 0.1])
    np.testing.assert_allclose(packed[1:, :3], 0)
    np.testing.assert_allclose(packed[1:, 3], -100)


def test_pack_attached_spheres_rejects_empty_wrong_type_and_overflow():
    with pytest.raises(ValueError):
        pack_attached_spheres((), 1)
    with pytest.raises(ValueError):
        pack_attached_spheres((object(),), 1)
    with pytest.raises(ValueError):
        pack_attached_spheres((AttachedSphere((0, 0, 0), 1),) * 2, 1)
