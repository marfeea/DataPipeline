"""Locate the external IK/Isaac runtime without importing it before AppLauncher."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def ensure_curriculumrl_available() -> None:
    """Prefer installed packages, then an explicit or workspace source checkout."""
    if importlib.util.find_spec("CurriculumRL") is not None:
        return
    configured = os.environ.get("DATAPIPE_CURRICULUMRL_ROOT")
    roots = (
        [Path(configured).expanduser()]
        if configured
        else [parent / "CurriculumRL" for parent in Path(__file__).resolve().parents]
    )
    for root in roots:
        for source in (root / "source" / "CurriculumRL", root):
            if (source / "CurriculumRL" / "__init__.py").is_file():
                sys.path.insert(0, str(source.resolve()))
                return
    raise ModuleNotFoundError(
        "DataPipe 配置已本地化，但 Isaac 采集/回放仍需要 CurriculumRL 的 IK 和环境。"
        "请用当前 Python 执行 pip install -e <CurriculumRL仓库>/source/CurriculumRL，"
        "或设置 DATAPIPE_CURRICULUMRL_ROOT 指向该仓库。",
        name="CurriculumRL",
    )
