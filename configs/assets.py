"""外部资产路径和 AUBO USD prim 契约的唯一事实源。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

ASSET_ROOT_ENV_VAR: Final = "CURRICULUMRL_ASSET_ROOT"


class AssetRootError(RuntimeError):
    """资产根目录未配置或无法自动发现。"""


@dataclass(frozen=True)
class AssetSpec:
    """仓库外部资产的相对路径契约。"""

    key: str
    relative_path: Path
    required: bool = True


@dataclass(frozen=True)
class RobotPrimContract:
    """带夹爪 AUBO USD 中必须保持稳定的 prim 与关节名称。"""

    usd_default_prim: str
    articulation_prim: str
    flange_body: str
    arm_joints: tuple[str, ...]
    gripper_joints: tuple[str, ...]
    ignored_contact_bodies: tuple[str, ...]
    contact_bodies: tuple[str, ...]


TEST_SUPPORT_TRAY_INDICES: Final = (1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14)

ASSETS: Final = (
    AssetSpec("aubo_with_gripper", Path("AUBO_E5/AUBO_E5_Withclaw.usd")),
    AssetSpec("workstation", Path("QKL-HX-300-II-00/Part/WorkStation/WorkStation.usd")),
    AssetSpec("sample_bottle", Path("QKL-HX-300-II-00/Part/Reagent_01/M_Reagent_01.usd")),
    AssetSpec("laboratory", Path("Laboratory/M_Laboratory.usd")),
    *(
        AssetSpec(
            f"support_tray_{index:02d}",
            Path(f"QKL-HX-300-II-00/Part/SupportTray_{index:02d}/M_SupportTray_{index:02d}.usd"),
        )
        for index in TEST_SUPPORT_TRAY_INDICES
    ),
)

ROBOT_PRIM_CONTRACT: Final = RobotPrimContract(
    usd_default_prim="Root",
    articulation_prim="AUBO_E5",
    flange_body="Flange",
    arm_joints=("Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Flange"),
    gripper_joints=("UpperFinger", "DownFinger"),
    ignored_contact_bodies=("Base_Link",),
    contact_bodies=("Base_Link", "Link_01", "Link_02", "Link_03", "Link_04", "Link_05", "Flange"),
)

SCENE_ENTITY_AUBO: Final = "AUBObot"
SCENE_ENTITY_AUBO_2: Final = "AUBObot_2"
SCENE_ENTITY_TARGET: Final = "ws_interactive_reagent_01_sample_bottle"
ROBOT_CONTACT_PRIM_EXPR: Final = "{ENV_REGEX_NS}/AUBObot/AUBO_E5/.*"


def _discover_workspace_asset_root() -> Path | None:
    """从当前扩展的祖先目录发现同级 ``Asset``，不固化机器绝对路径。"""

    module_path = Path(__file__).resolve()
    for parent in module_path.parents:
        candidate = parent.parent / "Asset"
        if candidate.is_dir():
            return candidate.resolve()
    return None


def resolve_asset_root(explicit_root: str | os.PathLike[str] | None = None) -> Path:
    """按显式参数、环境变量、工作区相邻目录的顺序解析资产根。"""

    if explicit_root is not None:
        root = Path(explicit_root).expanduser().resolve()
        source = "explicit argument"
    elif configured_root := os.environ.get(ASSET_ROOT_ENV_VAR):
        root = Path(configured_root).expanduser().resolve()
        source = ASSET_ROOT_ENV_VAR
    elif discovered_root := _discover_workspace_asset_root():
        root = discovered_root
        source = "workspace sibling discovery"
    else:
        raise AssetRootError(f"无法解析资产根目录；请设置环境变量 {ASSET_ROOT_ENV_VAR} 或传入显式路径。")

    if not root.is_dir():
        raise AssetRootError(f"资产根目录不存在（来源：{source}）：{root}")
    return root


def asset_path(spec: AssetSpec, explicit_root: str | os.PathLike[str] | None = None) -> Path:
    """返回资产解析后的绝对路径。"""

    return resolve_asset_root(explicit_root) / spec.relative_path


def asset_by_key(key: str) -> AssetSpec:
    """按稳定 key 查询资产定义。"""

    matches = [spec for spec in ASSETS if spec.key == key]
    if len(matches) != 1:
        raise KeyError(f"资产 key 必须唯一且存在：{key!r}，实际匹配 {len(matches)} 项")
    return matches[0]
