"""Shared-plan adapter preserves LingBot chunk cadence and EEF conventions."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'policy/LingBot_VA/lingbot_va/evaluation/robotwin/shared_plan_policy.py'
package = types.ModuleType('shared_plan_test_package')
package.__path__ = [str(PATH.parent)]
client_module = types.ModuleType('shared_plan_test_package.websocket_client_policy')
client_module.WebsocketClientPolicy = object
spec = importlib.util.spec_from_file_location('shared_plan_test_package.shared_plan_policy', PATH)
policy = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {package.__name__: package, client_module.__name__: client_module}):
    spec.loader.exec_module(policy)


class Client:
    def __init__(self, **kwargs): self.calls = []
    def infer(self, data):
        self.calls.append(data)
        if data.get('reset') or data.get('compute_kv_cache'): return {}
        action = np.zeros((16, 2, 16))
        action[6] = action[14] = 1  # xyzw identity rotations
        return {'action': action}


class Env:
    def __init__(self, limit=100):
        self.step_lim = limit
        self.take_action_cnt = 0
        self.eval_success = False
        self.actions = []
    def get_instruction(self): return 'frozen instruction'
    def get_obs(self):
        return dict(joint_action=dict(vector=np.zeros(14)),
                    endpose=dict(left_endpose=[1, 2, 3, 1, 0, 0, 0], left_gripper=0,
                                 right_endpose=[4, 5, 6, 1, 0, 0, 0], right_gripper=0),
                    observation={key: dict(rgb=np.full((2, 2, 3), i, dtype=np.uint8))
                                 for i, key in enumerate(['head_camera', 'left_camera', 'right_camera',
                                                         'head_camera_right', 'left_camera_right', 'right_camera_right'])})
    def take_action(self, action, action_type):
        self.actions.append(action)
        self.take_action_cnt += 1


class LingBotSharedPlanTests(unittest.TestCase):
    def make_policy(self, stereo=False):
        with patch.object(policy, 'WebsocketClientPolicy', Client):
            return policy.SharedPlanPolicy(host='localhost', port=1, stereo=stereo,
                                           video_guidance_scale=5, action_guidance_scale=1)

    def test_xyzw_training_identity_becomes_wxyz_robot_action(self):
        initial = np.array([1, 2, 3, 0, 0, 0, 1, 0] * 2)
        relative = np.array([.5, 0, 0, 0, 0, 0, 1, .7] * 2)
        np.testing.assert_allclose(policy.native_eef_action(relative, initial),
                                   [1.5, 2, 3, 1, 0, 0, 0, .7] * 2)

    def test_native_first_frame_skip_keyframes_and_cache_feedback(self):
        model, env = self.make_policy(), Env()
        model.step(env, env.get_obs())
        self.assertEqual(len(env.actions), 16)
        feedback = model.client.calls[-1]
        self.assertTrue(feedback['compute_kv_cache'])
        self.assertEqual(len(feedback['obs']), 4)
        self.assertEqual(feedback['state'].shape, (16, 2, 16))
        model.step(env, None)
        self.assertEqual(len(env.actions), 48)
        self.assertEqual(len(model.client.calls[-1]['obs']), 8)
        model.reset()
        self.assertTrue(model.should_request_observation())

    def test_common_step_limit_is_not_overshot(self):
        model, env = self.make_policy(), Env(limit=3)
        model.step(env, env.get_obs())
        self.assertEqual(len(env.actions), 3)
        self.assertFalse(model.client.calls[-1].get('compute_kv_cache', False))

    def test_stereo_uses_independent_right_images_and_missing_eye_fails(self):
        observation = Env().get_obs()
        encoded = policy.format_observation(observation, 'text', True)
        self.assertFalse(np.array_equal(encoded['observation.images.cam_high'],
                                        encoded['observation.images.cam_high_right']))
        del observation['observation']['head_camera_right']
        with self.assertRaises(KeyError): policy.format_observation(observation, 'text', True)
        self.assertNotIn('observation.images.cam_high_right', policy.format_observation(observation, 'text', False))

    def test_invalid_action_and_quaternion_fail(self):
        with self.assertRaises(ValueError): policy.native_eef_action(np.zeros(16), np.zeros(16))
        with self.assertRaises(ValueError): policy.native_eef_action(np.full(16, np.nan), np.zeros(16))
        model, env = self.make_policy(), Env()
        with patch.object(model.client, 'infer', return_value={'action': np.zeros((16, 2, 0))}):
            with self.assertRaises(ValueError): model.step(env, env.get_obs())


if __name__ == '__main__': unittest.main()
