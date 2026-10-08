"""Inspect AUBO USD rigid collision geometry and approximation settings."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from DataPipe.runtime import ensure_curriculumrl_available
ensure_curriculumrl_available()
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
launcher=AppLauncher(args)
from pxr import Usd, UsdPhysics, UsdGeom
from CurriculumRL.configs.assets import asset_path,asset_by_key

def main():
    path=asset_path(asset_by_key('aubo_with_gripper'))
    stage=Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(path)
    result=[]
    for prim in stage.Traverse():
        p=str(prim.GetPath())
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or not ('Link_02' in p or 'ClawTool_C' in p):
            continue
        mesh=UsdGeom.Mesh(prim)
        mesh_api=UsdPhysics.MeshCollisionAPI(prim)
        result.append({'path':p,'type':prim.GetTypeName(),
            'enabled':UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get(),
            'approximation':mesh_api.GetApproximationAttr().Get() if mesh_api else None,
            'vertices':len(mesh.GetPointsAttr().Get()) if mesh else None,
            'faces':len(mesh.GetFaceVertexCountsAttr().Get()) if mesh else None})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
try:
    main()
finally:
    launcher.app.close()
