"""Render a matched left/right LIBERO dataset at each recorded pre-action state."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import h5py
import mujoco
import numpy as np


def prepare_xml(xml, assets, robosuite):
    root = ET.fromstring(xml)
    for node in root.iter():
        old = node.get('file')
        if old is None:
            continue
        if '/robosuite/' in old:
            path = robosuite / old.rsplit('/robosuite/', 1)[1]
        elif '/assets/' in old:
            path = assets / old.split('/assets/', 1)[1]
        else:
            raise ValueError(f'Unknown asset reference: {old}')
        node.set('file', str(path.resolve(strict=True)))
    for name, baseline in [('agentview', .06), ('robot0_eye_in_hand', .02)]:
        matches = [(p, c) for p in root.iter() for c in p if c.tag == 'camera' and c.get('name') == name]
        if len(matches) != 1:
            raise ValueError(f'Camera not unique: {name}')
        parent, left = matches[0]
        quat = np.fromstring(left.get('quat', ''), sep=' ')
        if quat.shape != (4,) or left.get('mode', 'fixed') != 'fixed':
            raise ValueError('Expected fixed camera with quaternion')
        w, x, y, z = quat / np.linalg.norm(quat)
        axis = np.array([1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)])
        right = copy.deepcopy(left)
        right.set('name', name + '_stereo_right')
        right.set('pos', ' '.join(map(str, np.fromstring(left.get('pos'), sep=' ') + baseline*axis)))
        parent.append(right)
    return ET.tostring(root, encoding='unicode')


class FreshStereoEvalEnv:
    """Render evaluation observations with the admitted training camera recipe.

    Both VAE variants use the same fresh left images; only StereoTok receives
    the additional right images. Rendering never advances simulation time.
    """
    def __init__(self, *, stereo=False, **kwargs):
        from libero.libero.envs import OffScreenRenderEnv
        self._wrapped = OffScreenRenderEnv(**kwargs)
        self.stereo = stereo
        self.height = kwargs['camera_heights']
        self.width = kwargs['camera_widths']
        self.renderer = None

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def reset(self):
        from libero.libero import get_libero_path
        import robosuite
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        obs = self._wrapped.reset()
        env = self._wrapped.env
        state = env.sim.get_state()
        xml = prepare_xml(env.sim.model.get_xml(), Path(get_libero_path('assets')),
                          Path(robosuite.__file__).parent)
        env.reset_from_xml_string(xml)
        env.sim.set_state(state)
        env.sim.forward()
        self.renderer = mujoco.Renderer(env.sim.model._model, height=self.height, width=self.width)
        return self._images(obs)

    def _images(self, obs):
        env = self._wrapped.env
        mujoco.mj_forward(env.sim.model._model, env.sim.data._data)
        env._post_process()
        obs = dict(obs)
        obs.update(env._get_observations(force_update=True))
        for camera in ('agentview', 'robot0_eye_in_hand'):
            for suffix in ('', '_stereo_right') if self.stereo else ('',):
                self.renderer.update_scene(env.sim.data._data, camera=camera + suffix,
                    scene_option=env.sim._render_context_offscreen.vopt)
                obs[camera + suffix + '_image'] = self.renderer.render().copy()
        obs['fresh_rgb'] = True
        return obs

    def set_init_state(self, state):
        return self._images(self._wrapped.set_init_state(state))

    def step(self, action):
        obs, reward, done, info = self._wrapped.step(action)
        return self._images(obs), reward, done, info

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self._wrapped.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['source', 'output', 'libero-assets', 'robosuite-root']:
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--episode', help='One complete episode; omit for the complete task file')
    args = p.parse_args()
    from libero.libero.envs import TASK_MAPPING
    if args.output.exists() or Path(str(args.output) + '.partial').exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    receipts = []
    with h5py.File(args.source, 'r') as src, h5py.File(str(args.output) + '.partial', 'w') as dst:
        out = dst.create_group('data')
        for key, value in src['data'].attrs.items():
            out.attrs[key] = value
        out.attrs['macros_image_convention'] = 'opencv'
        out.attrs['observation_phase'] = 'recorded_state_before_corresponding_action'
        out.attrs['pairing'] = 'fresh-left+fresh-right'
        out.attrs['source_path'] = str(args.source.resolve())
        env_args = json.loads(src['data'].attrs['env_args'])
        kwargs = env_args['env_kwargs']
        kwargs['bddl_file_name'] = str((args.libero_assets.resolve().parents[2] / src['data'].attrs['bddl_file_name']).resolve(strict=True))
        env = TASK_MAPPING[env_args['problem_name']](**kwargs)
        try:
            names = [args.episode] if args.episode else sorted(src['data'], key=lambda v: int(v.split('_')[-1]))
            for name in names:
                episode = src['data'][name]
                states, actions = episode['states'][:], episode['actions'][:]
                if len(states) != len(actions):
                    raise ValueError('State/action lengths differ')
                env.reset()
                xml = prepare_xml(episode.attrs['model_file'], args.libero_assets, args.robosuite_root)
                env.reset_from_xml_string(xml)
                model, data = env.sim.model._model, env.sim.data._data
                spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
                if states.shape[1] != mujoco.mj_stateSize(model, spec):
                    raise ValueError('Recorded state shape mismatch')
                h, w = episode['obs/agentview_rgb'].shape[1:3]
                group = out.create_group(name)
                for key, value in episode.attrs.items():
                    group.attrs[key] = value
                group.attrs['model_file'] = xml
                group.create_dataset('states', data=states)
                group.create_dataset('actions', data=actions)
                obs = group.create_group('obs')
                pairs = [('agentview', 'agentview_rgb', .06), ('robot0_eye_in_hand', 'eye_in_hand_rgb', .02)]
                for left, key, baseline in pairs:
                    for suffix in ['', '_right']:
                        obs.create_dataset(key + suffix, (len(states), h, w, 3), dtype='u1', chunks=(1,h,w,3), compression='lzf')
                        obs.create_dataset(key + suffix + '_depth_z_m', (len(states), h, w), dtype='f4', chunks=(1,h,w), compression='lzf')
                        obs.create_dataset(key + suffix + '_T_world_camera_cv', (len(states),4,4), dtype='f8')
                    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, left)
                    focal = .5*h/np.tan(np.deg2rad(model.cam_fovy[cid])*.5)
                    obs.create_dataset(key + '_K', data=[[focal,0,w/2],[0,focal,h/2],[0,0,1]])
                proprio = {}
                renderer = mujoco.Renderer(model, height=h, width=w)
                try:
                    for index, state in enumerate(states):
                        mujoco.mj_setState(model, data, state, spec)
                        mujoco.mj_forward(model, data)
                        env._post_process()
                        current = env._get_observations(force_update=True)
                        import robosuite.utils.transform_utils as transform
                        values = dict(ee_pos=current['robot0_eef_pos'],
                                      ee_ori=transform.quat2axisangle(current['robot0_eef_quat']),
                                      gripper_states=current['robot0_gripper_qpos'], joint_states=current['robot0_joint_pos'])
                        values['ee_states'] = np.concatenate([values['ee_pos'], values['ee_ori']])
                        for key,value in values.items():
                            proprio.setdefault(key, []).append(np.asarray(value).copy())
                        before = np.empty_like(state)
                        mujoco.mj_getState(model, data, before, spec)
                        np.testing.assert_array_equal(before, state)
                        for left, key, baseline in pairs:
                            matrices = []
                            for suffix, camera in [('',left),('_right',left+'_stereo_right')]:
                                cid = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,camera)
                                matrix = np.eye(4)
                                matrix[:3,:3] = data.cam_xmat[cid].reshape(3,3) @ np.diag([1.,-1.,-1.])
                                matrix[:3,3] = data.cam_xpos[cid]
                                matrices.append(matrix)
                                obs[key+suffix+'_T_world_camera_cv'][index] = matrix
                                renderer.disable_depth_rendering()
                                renderer.update_scene(data,camera=camera,scene_option=env.sim._render_context_offscreen.vopt)
                                obs[key+suffix][index] = renderer.render()
                                renderer.enable_depth_rendering()
                                renderer.update_scene(data,camera=camera,scene_option=env.sim._render_context_offscreen.vopt)
                                depth = renderer.render().copy()
                                if not np.isfinite(depth).all() or np.any(depth <= 0):
                                    raise ValueError('Invalid metric depth')
                                obs[key+suffix+'_depth_z_m'][index] = depth
                            a,b = matrices
                            np.testing.assert_allclose(a[:3,:3],b[:3,:3],atol=1e-10,rtol=0)
                            np.testing.assert_allclose(a[:3,:3].T@(b[:3,3]-a[:3,3]),[baseline,0,0],atol=1e-10,rtol=0)
                        after = np.empty_like(state)
                        mujoco.mj_getState(model,data,after,spec)
                        np.testing.assert_array_equal(before,after)
                finally:
                    renderer.close()
                for key, values in proprio.items():
                    obs.create_dataset(key,data=np.asarray(values))
                receipts.append(dict(episode=name,frames=len(states),terminal_included=True,
                                     action_sha256=hashlib.sha256(actions.tobytes()).hexdigest()))
                dst.flush()
                print(json.dumps(receipts[-1]),flush=True)
            out.attrs['num_demos'] = len(names)
            out.attrs['total'] = sum(x['frames'] for x in receipts)
        finally:
            env.close()
    Path(str(args.output)+'.partial').rename(args.output)
    report = dict(source=str(args.source),output=str(args.output),episodes=receipts,
                  observation_phase='recorded_state_before_corresponding_action',
                  render_preserves_state=True,baselines_m=[.06,.02],elapsed_seconds=time.monotonic()-start,
                  native_uses='new left',stereotok_uses='same new left and new right')
    Path(str(args.output)+'.json').write_text(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
