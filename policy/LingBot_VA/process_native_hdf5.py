"""Prepare native LingBot latent-only LeRobot tables from admitted HDF5 inputs.

Original images remain in the source HDF5 files; no duplicate MP4 is produced.
Each parallel range is an independent dataset for MultiLatentLeRobotDataset.
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from process_data import (compute_column_stats, encode_text, encode_video_latent,
                          load_text_encoder, load_tokenizer, load_vae,
                          sample_frame_ids, WanVAEStreamingWrapper)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from XPolicyLab.utils.process_data import decode_image_bit


def enumerate_inputs(kind, source):
    if kind == 'robotwin':
        manifest = source / 'stereo-fill39-20261013/published/manifest.json'
        return [dict(path=str(source / e['path']), group='',
                     task=e['seen_instructions'][0], source_episode=e['source_episode_index'])
                for e in json.loads(manifest.read_text())['episodes']]
    rows = []
    for path in sorted(source.glob('*.hdf5')):
        with h5py.File(path) as f:
            task = json.loads(f['data'].attrs['problem_info'])['language_instruction']
            for group in sorted(f['data'], key=lambda x: int(x.split('_')[-1])):
                rows.append(dict(path=str(path), group='data/' + group,
                                 task=task, source_episode=group))
    return rows


def read_numeric(group, kind):
    if kind == 'libero':
        return (np.asarray(group['actions'], dtype=np.float32),
                np.concatenate([group['obs/ee_states'][:], group['obs/gripper_states'][:]], axis=1).astype(np.float32))
    arms = []
    for arm in ['left', 'right']:
        pose = group[f'endpose/{arm}_endpose'][:]
        arms.extend([pose[:, :3], pose[:, [4, 5, 6, 3]],
                     group[f'endpose/{arm}_gripper'][:].reshape(-1, 1)])
    action = np.concatenate(arms, axis=1).astype(np.float32)
    return action, action.copy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kind', choices=['robotwin', 'libero'], required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base-model', required=True)
    p.add_argument('--stereotok-checkpoint', help='Selected frozen u15853 export; reads matching right eyes')
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--count', type=int, required=True)
    args = p.parse_args()
    rows = enumerate_inputs(args.kind, args.source)[args.start:args.start + args.count]
    if len(rows) != args.count or args.start < 0 or args.count < 1:
        raise ValueError('Requested range is outside the source')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    (args.output / 'meta').mkdir()
    (args.output / 'source-episodes.json').write_text(json.dumps(rows, indent=2))
    vae = load_vae(str(Path(args.base_model) / 'vae'), torch.bfloat16, 'cuda',
                   stereotok_checkpoint=args.stereotok_checkpoint)
    wrapper = WanVAEStreamingWrapper(vae)
    tokenizer = load_tokenizer(str(Path(args.base_model) / 'tokenizer'))
    text_encoder = load_text_encoder(str(Path(args.base_model) / 'text_encoder'), torch.bfloat16, 'cuda')
    # Match author sampling: one raw step per LIBERO frame, four for RoboTwin.
    stride, fps = (4, 50) if args.kind == 'robotwin' else (1, 60)
    cameras = [('cam_high', 'head_camera', (256, 320)),
               ('cam_left_wrist', 'left_camera', (128, 160)),
               ('cam_right_wrist', 'right_camera', (128, 160))] if args.kind == 'robotwin' else [
                   ('agentview_rgb', 'agentview_rgb', (128, 128)),
                   ('eye_in_hand_rgb', 'eye_in_hand_rgb', (128, 128))]
    episodes, stats, tasks, text_cache = [], [], [], {}
    total = 0
    for index, row in enumerate(rows):
        with h5py.File(row['path']) as handle:
            group = handle[row['group']] if row['group'] else handle
            action, state = read_numeric(group, args.kind)
            n = len(action)
            if row['task'] not in tasks:
                tasks.append(row['task'])
                text_cache[row['task']] = encode_text(row['task'], tokenizer, text_encoder)
            chunk = f'chunk-{index // 1000:03d}'
            data = dict(episode_index=np.full(n, index, dtype=np.int64),
                        index=np.arange(total, total + n, dtype=np.int64),
                        frame_index=np.arange(n, dtype=np.int64),
                        task_index=np.full(n, tasks.index(row['task']), dtype=np.int64),
                        timestamp=np.arange(n, dtype=np.float32) / fps,
                        action=list(action), **{'observation.state': list(state)})
            dest = args.output / 'data' / chunk / f'episode_{index:06d}.parquet'
            dest.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(data).to_parquet(dest, index=False)
            frame_ids = sample_frame_ids(n, stride)
            valid = np.asarray(frame_ids) < n
            ids = np.minimum(frame_ids, n - 1)
            for key, source_key, size in cameras:
                right_frames = None
                if args.kind == 'robotwin':
                    images = group[f'observation/{source_key}/rgb']
                    frames = np.stack([decode_image_bit(images[int(i)]) for i in ids])
                    if args.stereotok_checkpoint:
                        right_images = group[f'observation/{source_key}_right/rgb']
                        right_frames = np.stack([decode_image_bit(right_images[int(i)]) for i in ids])
                else:
                    images = group[f'obs/{source_key}']
                    convention = handle['data'].attrs.get('macros_image_convention', 'opengl')
                    if convention not in ('opencv', 'opengl'):
                        raise ValueError(f'Unknown image convention: {convention}')
                    frames = np.stack([images[int(i)] for i in ids])
                    if convention == 'opengl':
                        frames = frames[:, ::-1].copy()
                    if args.stereotok_checkpoint:
                        if handle['data'].attrs.get('pairing') != 'fresh-left+fresh-right':
                            raise ValueError('StereoTok LIBERO requires admitted fresh paired data')
                        right_images = group[f'obs/{source_key}_right']
                        right_frames = np.stack([right_images[int(i)] for i in ids])
                        if convention == 'opengl':
                            right_frames = right_frames[:, ::-1].copy()
                if len(images) != n:
                    raise ValueError('Image/action length mismatch')
                if right_frames is not None and len(right_images) != n:
                    raise ValueError('Right image/action length mismatch')
                latent, f, h, w = encode_video_latent(frames, vae, wrapper, 'cuda', torch.bfloat16, size,
                                                     right_rgb=right_frames)
                rec = dict(latent=latent, latent_num_frames=f, latent_height=h, latent_width=w,
                           video_num_frames=len(frame_ids), video_height=size[0], video_width=size[1],
                           text_emb=text_cache[row['task']], text=row['task'], frame_ids=frame_ids,
                           frame_valid=valid.tolist(), start_frame=0, end_frame=n,
                           fps=fps / stride, ori_fps=fps,
                           tokenizer_binding=getattr(vae, 'binding', {'kind': 'official-wan'}))
                dest = args.output / 'latents' / chunk / ('observation.images.' + key)
                dest.mkdir(parents=True, exist_ok=True)
                torch.save(rec, dest / f'episode_{index:06d}_0_{n}.pth')
            episodes.append(dict(episode_index=index, tasks=[row['task']], length=n,
                                 action_config=[dict(start_frame=0, end_frame=n, action_text=row['task'])]))
            stats.append(dict(episode_index=index, stats={
                'action': compute_column_stats(action), 'observation.state': compute_column_stats(state)}))
            total += n
        print(f'episode={index} source={args.start + index} frames={n}', flush=True)
    torch.save(encode_text('', tokenizer, text_encoder), args.output / 'empty_emb.pt')
    features = {k: dict(dtype='int64', shape=[1], names=None)
                for k in ['episode_index', 'index', 'frame_index', 'task_index']}
    features.update(timestamp=dict(dtype='float32', shape=[1], names=None),
                    action=dict(dtype='float32', shape=[action.shape[1]], names=None))
    features['observation.state'] = dict(dtype='float32', shape=[state.shape[1]], names=None)
    info = dict(codebase_version='v2.1', robot_type='aloha' if args.kind == 'robotwin' else 'Franka',
                total_episodes=len(rows), total_frames=total, total_tasks=len(tasks), total_videos=0,
                total_chunks=(len(rows) + 999) // 1000, chunks_size=1000, fps=fps,
                splits={'train': f'0:{len(rows)}'},
                data_path='data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
                features=features)
    (args.output / 'meta/info.json').write_text(json.dumps(info, indent=2))
    for name, items in [('episodes', episodes), ('episodes_stats', stats),
                        ('tasks', [dict(task_index=i, task=t) for i, t in enumerate(tasks)])]:
        (args.output / f'meta/{name}.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in items))
    print('CONVERSION_COMPLETE', args.output, flush=True)


if __name__ == '__main__':
    main()
