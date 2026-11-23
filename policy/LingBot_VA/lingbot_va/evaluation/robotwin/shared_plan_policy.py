"""Native LingBot EEF inference connected to the common frozen-plan simulator."""
import time
import numpy as np
from scipy.spatial.transform import Rotation

from .websocket_client_policy import WebsocketClientPolicy


def format_observation(observation, prompt, stereo):
    result = {'task': prompt, 'observation.state': observation['joint_action']['vector']}
    for target, camera in [('cam_high', 'head_camera'), ('cam_left_wrist', 'left_camera'),
                           ('cam_right_wrist', 'right_camera')]:
        result['observation.images.' + target] = observation['observation'][camera]['rgb']
        if stereo:
            result['observation.images.' + target + '_right'] = observation['observation'][camera + '_right']['rgb']
    return result


def native_eef_action(relative, initial):
    """Training stores relative xyz and xyzw; RoboTwin takes absolute xyz/wxyz."""
    relative = np.asarray(relative)
    if relative.shape != (16,) or not np.isfinite(relative).all():
        raise ValueError('Expected finite LingBot RoboTwin 16D EEF action')
    arms = []
    for offset in (0, 8):
        action, anchor = relative[offset:offset + 8], initial[offset:offset + 8]
        rotation = (Rotation.from_quat(anchor[3:7]) * Rotation.from_quat(action[3:7])).as_quat()
        arms.extend([*(action[:3] + anchor[:3]), *rotation[[3, 0, 1, 2]], action[7]])
    return np.asarray(arms)


class SharedPlanPolicy:
    def __init__(self, *, host, port, stereo, video_guidance_scale, action_guidance_scale):
        if type(stereo) is not bool:
            raise ValueError('stereo must explicitly match the checkpoint encoder')
        self.client = WebsocketClientPolicy(host=host, port=port)
        self.stereo = stereo
        self.video_guidance_scale = video_guidance_scale
        self.action_guidance_scale = action_guidance_scale
        self.inference_seconds = []
        self.reset()

    def reset(self):
        self.first = True
        self.initial = None
        self.first_observation = None
        self.inference_seconds.clear()

    def should_request_observation(self):
        return self.first

    def step(self, env, observation):
        prompt = env.get_instruction()
        if self.first:
            self.client.infer(dict(reset=True, prompt=prompt, save_visualization=False))
            endpose = observation['endpose']
            initial = []
            for arm in ('left', 'right'):
                pose = np.asarray(endpose[arm + '_endpose'])
                initial.extend([*pose[:3], *pose[[4, 5, 6, 3]], endpose[arm + '_gripper']])
            self.initial = np.asarray(initial)
            self.first_observation = format_observation(observation, prompt, self.stereo)
        started = time.perf_counter()
        result = self.client.infer(dict(obs=self.first_observation, prompt=prompt,
            save_visualization=False, video_guidance_scale=self.video_guidance_scale,
            action_guidance_scale=self.action_guidance_scale))
        self.inference_seconds.append(time.perf_counter() - started)
        action = np.asarray(result['action'])
        if (action.ndim != 3 or action.shape[0] != 16 or action.shape[2] == 0
                or action.shape[2] % 4 or not np.isfinite(action).all()):
            raise ValueError('Expected finite LingBot action [16, frames, steps], steps divisible by four')
        start = 1 if self.first else 0
        if action.shape[1] <= start:
            raise ValueError('LingBot initial-frame skip would leave an empty action chunk')
        frames = []
        stride = action.shape[2] // 4
        for i in range(start, action.shape[1]):
            for j in range(action.shape[2]):
                env.take_action(native_eef_action(action[:, i, j], self.initial), action_type='ee')
                if (j + 1) % stride == 0:
                    frames.append(format_observation(env.get_obs(), prompt, self.stereo))
                if env.eval_success or env.take_action_cnt >= env.step_lim:
                    self.first = False
                    return
        self.first = False
        self.client.infer(dict(obs=frames, compute_kv_cache=True, imagine=False,
                               save_visualization=False, state=action))
