"""Inventory successful recordings, then render every source sample in a visible GUI."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def write(p, v):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(v, ensure_ascii=False, indent=2), encoding='utf-8')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def inventory():
    found, excluded, seen = [], [], {}
    archive = ROOT/'data/curobo_pick_place_graph_run003_success_20261004'
    candidates = list(archive.rglob('trajectory.npz'))
    candidates += list((ROOT/'data').rglob('trajectory.npz'))
    candidates += list((ROOT.parent/'artifacts').rglob('trajectory.npz'))
    visited = set()
    for p in candidates:
        if p in visited: continue
        visited.add(p)
        if any(s in str(p).lower() for s in ('pytest', 'test_artifact', 'test_append', 'test_npz', 'implementation_')): continue
        rp = p.parent/'plan_report.json'
        kind = 'hybrid'
        if not rp.exists():
            kind = 'measured'
            rp = p.parent/'report.json'
            if not rp.exists(): rp = Path(str(rp).replace('trajectorys', 'trajdata'))
        if not rp.exists(): continue
        r = read(rp)
        schema = r.get('schema', '')
        if kind == 'hybrid' and schema != 'curobo_pick_place_v2_plan_report/1': continue
        if kind == 'measured' and schema not in ('physical_failure_experiment/1', 'curobo_v2_legacy_contact_isaac_probe/1'): continue
        if not r.get('passed') or r.get('accepted_for_use') is False or r.get('task_succeeded') is False:
            excluded.append(dict(source=str(p), reason='not a successful accepted task', report=str(rp)))
            continue
        snapshot, trace = p.parent/'scene_snapshot.json', p.parent/'trace.json'
        if not snapshot.exists() or kind == 'measured' and not trace.exists():
            excluded.append(dict(source=str(p), reason='missing state replay sidecars'))
            continue
        digest = sha(p)
        # Same bytes with a different scene/trace remain distinct recordings.
        identity = digest + sha(snapshot) + (sha(trace) if kind == 'measured' else '')
        if identity in seen:
            seen[identity]['equivalent_sources'].append(str(p))
            continue
        rel = p.parent.relative_to(ROOT.parent).as_posix()
        label = rel.replace('/', '__')
        item = dict(id=label, kind=kind, source=str(p), report=str(rp), snapshot=str(snapshot),
                    trace=str(trace) if trace.exists() else None, sha256=digest, equivalent_sources=[])
        found.append(item); seen[identity] = item
    return dict(schema='success_video_inventory/1', items=found, excluded=excluded)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', default='isaacsim')
    parser.add_argument('--run-id', default='success_1080p60_20261111_final')
    parser.add_argument('--inventory-only', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--pause-after-current', action='store_true')
    parser.add_argument('--source-filter', help='Select source paths containing this exact substring')
    args = parser.parse_args()
    if '--backend' not in sys.argv:
        sys.argv.extend(['--backend', 'isaacsim'])
    args.headless = False
    args.device = 'cuda:0'
    args.kit_cache_dir = ROOT/'data/isaacsim_kit_cache'
    output = ROOT/'data/test/trajdata'/args.run_id
    links = ROOT/'data/test/trajectorys'/args.run_id
    output.mkdir(parents=True, exist_ok=True)
    inv = inventory()
    if args.source_filter:
        inv['items'] = [x for x in inv['items'] if args.source_filter in x['source']]
        if not inv['items']: raise ValueError('No matching successful recordings')
    write(output/'inventory.json', inv)
    print('INVENTORY', len(inv['items']), 'excluded', len(inv['excluded']), flush=True)
    if args.inventory_only: return
    import numpy as np
    import imageio_ffmpeg
    from PIL import Image
    runtime = Path(sys.executable).parent
    os.environ['PATH'] = os.pathsep.join([str(runtime), str(runtime/'Scripts'), str(runtime/'Library/bin'), os.environ.get('PATH','')])
    spec = importlib.util.spec_from_file_location('shared_runtime', ROOT/'testscripts/legacy_lift_speed_20261015/run_sweep.py')
    base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
    cache = base.install_shared_runtime(output, camera_warmup_timeout_s=120)
    from scripts import sim_backend
    launcher = sim_backend.launch(args)
    import torch
    from configs.assets import ROBOT_PRIM_CONTRACT
    from scripts.pipelines.curobo_bootstrap.robot_adapter import pose_matrix
    from scripts.pipelines.curobo_bootstrap.replay import matrix_pose_wxyz
    from scripts.pipelines.curobo_bootstrap.source_target import set_source_target_pose
    from scripts.pipelines.curobo_bootstrap.contact_timing import configure_contact_timing
    Env, Cfg = sim_backend.trajectory_classes()
    from scripts.native import scene
    # The stock missing-geometry heuristic samples the ceiling of a square
    # camera. A 16:9 crop has no ceiling there; retain a diagnostic and test
    # the full frame for sufficient neutral scene/robot pixels instead.
    def wide_scene_missing(owner):
        camera = owner.cameras.get('CameraSensor_3')
        if camera is None: return False
        rgb = camera.camera.get_rgb()
        if rgb is None or not np.asarray(rgb).size: return True
        rgb = np.asarray(rgb)[::16,::16,:3]
        if not (output/'native_initial_front.png').exists():
            Image.fromarray(rgb).save(output/'native_initial_front.png')
        spread = rgb.max(axis=2).astype(float)-rgb.min(axis=2)
        return float(np.mean((spread<30) & (rgb.mean(axis=2)>35)))<0.05
    scene.NativeTrajectoryEnv.background_only_render = wide_scene_missing
    import time
    def checked_fast_render(adapter):
        started=time.monotonic()
        while True:
            adapter.forward(); adapter.owner.world.render()
            adapter.owner.scene.update(adapter.get_physics_dt())
            if not adapter.owner.cameras_ready: return
            good=all(c.camera.get_rgb() is not None and np.asarray(c.camera.get_rgb())[::16,::16,:3].std()>5 for c in adapter.owner.cameras.values())
            if good and not adapter.owner.background_only_render(): return
            if time.monotonic()-started>120: raise RuntimeError('Native RGB unavailable')
    scene.SimAdapter.render=checked_fast_render
    cfg = Cfg(); cfg.sim.device = args.device
    configure_contact_timing(cfg, physics_hz=120, control_hz=60, velocity_iterations=4)
    cfg, _ = sim_backend.configure_cameras(cfg, 1920, 1080)
    cfg.camera_diagnostics_directory = output/'camera_warmup'
    env = Env(cfg); env.reset()
    for camera in env.cameras.values():
        camera.camera.set_focal_length(camera.camera.get_focal_length()*1080/1920)
    env.sim.render()
    import carb
    carb.settings.get_settings().set('/log/level', 'error')
    robot, target = env.robot, env.target
    arm, _ = robot.find_joints(ROBOT_PRIM_CONTRACT.arm_joints)
    fingers, _ = robot.find_joints(ROBOT_PRIM_CONTRACT.gripper_joints)
    IK = sim_backend.ik_class()
    ik = IK(robot, cfg.sim.dt, cfg.decimation, diagnostics_enabled=True, flange_downward_enabled=True)
    def tensor(x): return torch.as_tensor(x, dtype=robot.data.joint_pos.dtype, device=args.device)
    def cpu(x): return x.detach().cpu().numpy()
    base_pose = pose_matrix(cpu(robot.data.root_pos_w[0]), cpu(robot.data.root_quat_w[0]))
    def forbidden(*a, **kw): raise RuntimeError('Physics stepping forbidden during video state replay')
    env.sim.step = forbidden
    results = []
    try:
        for number, item in enumerate(inv['items'][:args.limit]):
            dest = output/item['id']
            if (dest/'video_report.json').exists() and read(dest/'video_report.json').get('passed'):
                results.append(read(dest/'video_report.json')); continue
            dest.mkdir(parents=True, exist_ok=True)
            report = dict(id=item['id'], passed=False, mode='visible_gui_state_replay',
                          source=item['source'], source_sha256=item['sha256'], width=1920, height=1080,
                          fps=60, physics_steps=0, videos=[], image_count=0)
            writers = {}; logfiles = []; frames = []
            print('START', number+1, '/', len(inv['items']), item['id'], flush=True)
            try:
                p = Path(item['source']); r = read(item['report']); snapshot = read(item['snapshot'])
                task = snapshot.get('task', snapshot)
                assert task['task_hash'] == r['task_hash']
                input_hashes={str(f):sha(f) for f in [p,Path(item['report']),Path(item['snapshot'])]+([Path(item['trace'])] if item['trace'] else [])}
                with np.load(p, allow_pickle=False) as data: arrays = {k:data[k].copy() for k in data.files}
                times = arrays['time_s']; n = len(times)
                np.testing.assert_allclose(np.diff(times), 1/60, atol=1e-6, rtol=0)
                np.testing.assert_allclose(base_pose, task['frames']['T_WB'], atol=1e-5, rtol=0)
                set_source_target_pose(env, task)
                rows = read(item['trace']) if item['kind']=='measured' else None
                if rows is not None:
                    assert len(rows)==n
                    np.testing.assert_allclose(arrays['joint_position'], [x['q'] for x in rows], atol=1e-7)
                else:
                    measured = r['measured']
                    lift_tcp = base_pose @ pose_matrix(measured['lift_tcp_pose_b'][:3], measured['lift_tcp_pose_b'][3:])
                    tcp_object = np.linalg.inv(lift_tcp) @ pose_matrix(measured['target_pose_w_after_lift'][:3], measured['target_pose_w_after_lift'][3:])
                    d = task['destination']
                    destination_pose = matrix_pose_wxyz(np.asarray(task['frames']['T_WE']) @ pose_matrix(d['position_e'],d['rotation_wxyz']))
                    source_pose = snapshot['target_pose_w_at_reset']
                write(links/item['id']/'replay_source.json',dict(**item, video_directory=str(dest)))
                for name in env.cameras:
                    path = dest/(name+'.mp4')
                    logfile = open(dest/(name+'_encoder.log'),'wb'); logfiles.append(logfile)
                    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r','60','-i','-','-an','-c:v','libx264','-threads','2','-preset','ultrafast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
                    writers[name] = subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=logfile)
                max_error = max_object_error = max_finger_error = max_object_rotation_error = 0
                for index in range(n):
                    if not launcher.app.is_running(): raise RuntimeError('GUI closed')
                    q, dq = robot.data.joint_pos.clone(), robot.data.joint_vel.clone()
                    q[:,arm], dq[:,arm] = tensor(arrays['joint_position'][index]), tensor(arrays['joint_velocity'][index])
                    q[:,fingers] = tensor(rows[index]['gripper_position'] if rows is not None else arrays['gripper_target'][index])
                    dq[:,fingers] = tensor(rows[index].get('gripper_velocity',[0,0]) if rows is not None else [0,0])
                    robot.write_joint_state_to_sim(q,dq); robot.set_joint_position_target(q, joint_ids=list(range(q.shape[-1])))
                    env.scene.write_data_to_sim(); env.sim.forward(); env.scene.update(cfg.sim.dt)
                    if rows is not None:
                        obj = rows[index]['target_pose_w']; stage = rows[index]['stage']
                    else:
                        stage = str(arrays['event_code'][index])
                        if arrays['object_attached'][index]:
                            tp,tq,_ = ik.current_tcp_pose_b()
                            obj = matrix_pose_wxyz(base_pose @ pose_matrix(cpu(tp[0]),cpu(tq[0])) @ tcp_object)
                        elif stage=='PLACE' or index>r['trajectory_acceptance']['place_frame']: obj=destination_pose
                        else: obj=source_pose
                    target.write_root_pose_to_sim(tensor(obj)[None]); target.write_root_velocity_to_sim(tensor([0]*6)[None])
                    env.scene.write_data_to_sim(); env.sim.forward(); env.scene.update(cfg.sim.dt); env.sim.render()
                    err=float(np.max(np.abs(cpu(robot.data.joint_pos[0,arm])-arrays['joint_position'][index])))
                    max_error=max(max_error,err)
                    if rows is not None:
                        max_object_error=max(max_object_error,float(np.linalg.norm(cpu(target.data.root_pos_w[0])-np.asarray(obj[:3]))))
                        max_finger_error=max(max_finger_error,float(np.max(np.abs(cpu(robot.data.joint_pos[0,fingers])-rows[index]['gripper_position']))))
                        actual=cpu(target.data.root_quat_w[0]).astype(float); expected=np.asarray(obj[3:],dtype=float)
                        cosine=abs(float(np.dot(actual/np.linalg.norm(actual),expected/np.linalg.norm(expected))))
                        max_object_rotation_error=max(max_object_rotation_error,float(2*np.arccos(np.clip(cosine,-1,1))))
                    for name,camera in env.cameras.items():
                        rgb=np.ascontiguousarray(camera.camera.get_rgb()[...,:3],dtype=np.uint8)
                        assert rgb.shape==(1080,1920,3) and rgb[::16,::16].std()>5
                        writers[name].stdin.write(rgb.tobytes())
                        # Preserve original three-view evidence and exact frame association.
                        changed = rows is not None and (index==0 or rows[index]['stage']!=rows[index-1]['stage'])
                        if index%12==0 or index==n-1 or changed or stage!='NONE' and rows is None:
                            img=dest/'frames'/name/f'{index:06d}.png'; img.parent.mkdir(parents=True,exist_ok=True)
                            Image.fromarray(rgb).save(img,compress_level=1)
                            frames.append(dict(camera=name,trajectory_index=index,time_s=float(times[index]),stage=stage,rgb=str(img.relative_to(dest))))
                    if (index+1)%120==0: print('FRAME',number+1,index+1,'/',n,flush=True)
                for name,writer in writers.items():
                    writer.stdin.close(); assert writer.wait()==0
                    import cv2
                    video=dest/(name+'.mp4'); cap=cv2.VideoCapture(str(video))
                    metadata=dict(path=str(video),width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),fps=cap.get(cv2.CAP_PROP_FPS),frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                    assert (metadata['width'],metadata['height'],metadata['fps'],metadata['frames'])==(1920,1080,60.,n),metadata
                    for sample in (0,n//2,n-1):
                        cap.set(cv2.CAP_PROP_POS_FRAMES,sample); ok,pixels=cap.read(); assert ok and pixels.std()>5
                    cap.release(); report['videos'].append(metadata)
                assert max_error<1e-5 and max_object_error<1e-5 and max_finger_error<1e-5 and max_object_rotation_error<1e-3
                assert all(sha(f)==digest for f,digest in input_hashes.items())
                report.update(passed=True,source_frames=n,duration_s=n/60,max_joint_error_rad=max_error,image_count=len(frames),
                              max_object_position_error_m=max_object_error,max_finger_error_m=max_finger_error,
                              max_object_rotation_error_rad=max_object_rotation_error,input_sha256=input_hashes,
                              object_pose_source='recorded target_pose_w per sample' if rows is not None else 'reconstructed logical attachment',
                              new_physics_validation=False,camera_focal_length_scale=1080/1920)
            except Exception as exc:
                report['failure_reason']=repr(exc); traceback.print_exc()
            finally:
                for writer in writers.values():
                    if writer.poll() is None:
                        try: writer.stdin.close(); writer.wait(timeout=30)
                        except Exception: writer.terminate()
                for logfile in logfiles: logfile.close()
                write(dest/'frames.json',dict(frames=frames))
                write(dest/'video_report.json',report); results.append(report)
                write(output/'summary.json',dict(total=len(inv['items']),completed=sum(x['passed'] for x in results),results=results))
                print('COMPLETE',number+1,report['passed'],report.get('failure_reason',''),flush=True)
            if args.pause_after_current or (output/'pause_after_current').exists():
                break
    finally:
        if 'real_close' in cache: cache['real_close']()
        launcher.app.close()

if __name__=='__main__': main()
