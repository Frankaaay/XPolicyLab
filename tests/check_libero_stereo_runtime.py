"""Real LIBERO renderer acceptance; no model score."""
import argparse
import json
import numpy as np
import mujoco
from libero.libero import benchmark
from XPolicyLab.utils.render_libero_stereo import FreshStereoEvalEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--resolution', type=int, required=True)
    args = p.parse_args()
    suite = benchmark.get_benchmark_dict()['libero_spatial']()
    saved = []
    for stereo in [False, True]:
        np.random.seed(42)
        env = FreshStereoEvalEnv(stereo=stereo,
            bddl_file_name=suite.get_task_bddl_file_path(0),
            camera_heights=args.resolution, camera_widths=args.resolution)
        try:
            env.seed(42)
            env.reset()
            obs = env.set_init_state(suite.get_task_init_states(0)[0])
            for _ in range(5):
                obs, _, _, _ = env.step([0.] * 7)
            native = env.env
            state = native.sim.get_state().flatten().copy()
            repeat = env._images(obs)
            np.testing.assert_array_equal(state, native.sim.get_state().flatten())
            for key in ['agentview_image', 'robot0_eye_in_hand_image']:
                np.testing.assert_array_equal(obs[key], repeat[key])
            saved.append(obs)
            if stereo:
                model, data = native.sim.model._model, native.sim.data._data
                for cam, baseline in [('agentview', .06), ('robot0_eye_in_hand', .02)]:
                    left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
                    right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam+'_stereo_right')
                    np.testing.assert_allclose(data.cam_xmat[left], data.cam_xmat[right], atol=1e-10, rtol=0)
                    np.testing.assert_allclose(data.cam_xmat[left].reshape(3,3).T @ (data.cam_xpos[right]-data.cam_xpos[left]), [baseline,0,0], atol=1e-10, rtol=0)
                    assert not np.array_equal(obs[cam+'_image'],obs[cam+'_stereo_right_image'])
        finally:
            env.close()
    for key in ['agentview_image', 'robot0_eye_in_hand_image']:
        np.testing.assert_array_equal(saved[0][key], saved[1][key])
    print('LIBERO_FRESH_STEREO_RUNTIME_PASSED', json.dumps({'resolution':args.resolution,'left_exact':True,'state_unchanged':True,'baselines_m':[.06,.02]}),flush=True)

if __name__ == '__main__':
    main()
