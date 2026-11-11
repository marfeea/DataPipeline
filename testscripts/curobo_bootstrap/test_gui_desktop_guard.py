import os
from types import SimpleNamespace

import pytest

from scripts import sim_backend
from scripts.native import gui_desktop


@pytest.mark.skipif(os.name != "nt", reason="Windows desktop contract")
def test_gui_launcher_rejects_an_isolated_desktop_before_loading_isaac(monkeypatch, tmp_path):
    monkeypatch.setattr(gui_desktop, "desktop_evidence", lambda: dict(desktop="CodexSandboxDesktop-test"))
    args = SimpleNamespace(backend="isaacsim", headless=False, kit_cache_dir=tmp_path / "cache")
    with pytest.raises(RuntimeError, match="Default desktop"):
        sim_backend.launch(args)
