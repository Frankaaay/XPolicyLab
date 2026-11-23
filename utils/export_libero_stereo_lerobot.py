"""Export admitted paired LIBERO HDF5 into existing FastWAM v2.1/OpenWAM v3 layouts."""
import argparse
import json
from pathlib import Path
from fractions import Fraction

import av
import h5py
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


CAMERAS = {'image': 'agentview_rgb', 'wrist_image': 'eye_in_hand_rgb',
           'image_right': 'agentview_rgb_right', 'wrist_image_right': 'eye_in_hand_rgb_right'}


def stats(x):
    return {k: v.tolist() for k, v in dict(min=x.min(0), max=x.max(0), mean=x.mean(0),
             std=x.std(0), count=np.array([len(x)])).items()}


def video(images, path, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('libx264', rate=Fraction(str(fps)))
        stream.height, stream.width = images.shape[1:3]
        stream.pix_fmt = 'yuv444p'
        stream.thread_count = 1
        stream.options = {'crf': '0', 'preset': 'fast'}
        for i in range(len(images)):
            frame = av.VideoFrame.from_ndarray(images[i], format='rgb24')
            frame.pts = i
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def export(source, output):
    if output.exists():
        raise FileExistsError(output)
    fast, openwam = output / 'fastwam', output / 'openwam'
    for root in [fast, openwam]:
        (root / 'meta').mkdir(parents=True)
    episodes, v3episodes, tasks, episode_stats, provenance = [], [], [], [], []
    offset, fps = 0, None
    for path in sorted(source.glob('*.hdf5')):
        with h5py.File(path) as h5:
            attrs = h5['data'].attrs
            if attrs.get('pairing') != 'fresh-left+fresh-right' or attrs.get('macros_image_convention') != 'opencv':
                raise ValueError('Only admitted opencv fresh stereo data can be exported')
            current_fps = json.loads(attrs['env_args'])['env_kwargs']['control_freq']
            if fps is not None and fps != current_fps:
                raise ValueError('Mixed control rates')
            fps = current_fps
            task = json.loads(attrs['problem_info'])['language_instruction']
            if task not in tasks:
                tasks.append(task)
            for name in sorted(h5['data'], key=lambda x: int(x.split('_')[-1])):
                group = h5['data'][name]
                action = group['actions'][:].astype(np.float32)
                state = np.concatenate([group['obs/ee_states'][:], group['obs/gripper_states'][:]], 1).astype(np.float32)
                n, index = len(action), len(episodes)
                chunk = index // 1000
                fields = dict(index=np.arange(offset, offset+n), episode_index=np.full(n,index),
                              frame_index=np.arange(n), timestamp=np.arange(n,dtype=np.float32)/fps,
                              task_index=np.full(n,tasks.index(task)))
                data = pd.DataFrame(dict(fields, action=list(action), **{'observation.state':list(state)}))
                dest = fast / f'data/chunk-{chunk:03d}/episode_{index:06d}.parquet'
                dest.parent.mkdir(parents=True, exist_ok=True)
                data.to_parquet(dest,index=False)
                action_rotation = Rotation.from_rotvec(action[:,3:6]).as_matrix()[:,:,:2].transpose(0,2,1).reshape(n,6)
                state_rotation = Rotation.from_rotvec(state[:,3:6]).as_matrix()[:,:,:2].transpose(0,2,1).reshape(n,6)
                action10 = np.concatenate([action[:,:3],action_rotation,-action[:,6:7]],1).astype(np.float32)
                grip = np.clip(2*(state[:,6:7]-state[:,7:8])/.08-1,-1,1)
                state10 = np.concatenate([state[:,:3],state_rotation,grip],1).astype(np.float32)
                dest = openwam / f'data/chunk-{chunk:03d}/file-{index:06d}.parquet'
                dest.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(dict(fields,action=list(action10),**{'observation.state':list(state10)})).to_parquet(dest,index=False)
                ep = dict(episode_index=index,length=n,tasks=[task])
                v3 = dict(ep, **{'data/chunk_index':chunk,'data/file_index':index,
                                'dataset_from_index':offset,'dataset_to_index':offset+n})
                features = {}
                for key, h5key in CAMERAS.items():
                    images = group['obs/'+h5key]
                    if len(images) != n:
                        raise ValueError('Image/action length mismatch')
                    cam='observation.images.'+key
                    dest=fast/f'videos/chunk-{chunk:03d}/{cam}/episode_{index:06d}.mp4'
                    video(images,dest,fps)
                    linked=openwam/f'videos/{cam}/chunk-{chunk:03d}/file-{index:06d}.mp4'
                    linked.parent.mkdir(parents=True,exist_ok=True)
                    linked.symlink_to(dest.resolve())
                    v3.update({f'videos/{cam}/chunk_index':chunk,f'videos/{cam}/file_index':index,
                               f'videos/{cam}/from_timestamp':0.,f'videos/{cam}/to_timestamp':n/fps})
                    features[cam]=dict(dtype='video',shape=list(images.shape[1:]),names=['height','width','channel'],
                                       info={'video.fps':fps,'video.height':images.shape[1],
                                             'video.width':images.shape[2],'video.channels':3,
                                             'video.codec':'h264','video.pix_fmt':'yuv444p','has_audio':False})
                episodes.append(ep);v3episodes.append(v3)
                episode_stats.append(dict(episode_index=index,stats={'action':stats(action),'observation.state':stats(state)}))
                provenance.append(dict(path=str(path),group=name,frames=n))
                offset+=n
                print(json.dumps(dict(episode=index,frames=n)),flush=True)
    if not episodes:
        raise ValueError('No episodes')
    numeric={k:dict(dtype='int64',shape=[1],names=None) for k in ['index','episode_index','frame_index','task_index']}
    numeric['timestamp']=dict(dtype='float32',shape=[1],names=None)
    for root,version,action_dim,state_dim in [(fast,'v2.1',7,8),(openwam,'v3.0',10,10)]:
        info=dict(codebase_version=version,robot_type='Franka',fps=fps,total_episodes=len(episodes),
                  total_frames=offset,total_tasks=len(tasks),total_videos=4*len(episodes),
                  total_chunks=(len(episodes)+999)//1000,chunks_size=1000,
                  splits={'train':f'0:{len(episodes)}'},features=dict(numeric,**features))
        info['features'].update(action=dict(dtype='float32',shape=[action_dim],names=None),
                               **{'observation.state':dict(dtype='float32',shape=[state_dim],names=None)})
        info.update(data_path='data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet' if root==fast else 'data/chunk-{chunk_index:03d}/file-{file_index:06d}.parquet',
                    video_path='videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4' if root==fast else 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:06d}.mp4')
        (root/'meta/info.json').write_text(json.dumps(info,indent=2))
        (root/'meta/stereo_provenance.json').write_text(json.dumps(dict(pairing='fresh-left+fresh-right',source_episodes=provenance,
            source_episode_indices=list(range(len(episodes))),source_total_episodes=len(episodes)),indent=2))
    for name,items in [('episodes',episodes),('episodes_stats',episode_stats),('tasks',[dict(task_index=i,task=t) for i,t in enumerate(tasks)])]:
        (fast/f'meta/{name}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in items))
    (openwam/'meta/episodes/chunk-000').mkdir(parents=True)
    pd.DataFrame(v3episodes).to_parquet(openwam/'meta/episodes/chunk-000/file-000000.parquet',index=False)
    pd.DataFrame({'task_index':range(len(tasks))},index=tasks).to_parquet(openwam/'meta/tasks.parquet')
    (openwam/'meta/conversion.json').write_text(json.dumps(dict(output_representation='native_delta_eef10',source=str(source),fps=fps)))
    (output/'complete.json').write_text(json.dumps(dict(episodes=len(episodes),frames=offset,fps=fps)))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    export(args.source,args.output)
