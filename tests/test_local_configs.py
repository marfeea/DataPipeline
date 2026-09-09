"""Local configuration must work without the external simulation runtime."""

import os
from pathlib import Path
import subprocess
import sys


def test_all_configs_import_without_curriculumrl():
    source = Path(__file__).resolve().parents[1]
    script = """
import importlib
import importlib.abc
import pkgutil
import sys

class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'CurriculumRL':
            raise AssertionError('Local configs imported the external runtime')

sys.meta_path.insert(0, BlockRuntime())
import configs
for module in pkgutil.iter_modules(configs.__path__):
    importlib.import_module('configs.' + module.name)
from DataPipe.camera_config import project_diagnostic_camera_sensor_specs
assert len(project_diagnostic_camera_sensor_specs()) == 3
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": str(source)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
