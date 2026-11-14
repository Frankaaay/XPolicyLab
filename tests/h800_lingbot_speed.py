"""Single idle GPU LingBot diagnostic; frozen scenes and exact paired traces."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import types

B = Path('/mnt/workdir/frank')
P = B/'projects/shared-robotwin-20261023'
F = P/'FastWAM'
OLD = P/'XPolicyLab'
NEW = Path(__file__).resolve().parents[1]
REL = Path('policy/LingBot_VA/lingbot_va/wan_va')
T = B/'runs/robotwin-u15853-mono-20261101'
CK = T/'lingbot/checkpoints/checkpoint_step_50000'
SPY = B/'envs/lingbot-va-baselines-20261014/bin/python'
CPY = B/'envs/fastwam-stereotok-eval-cu128-20261009/bin/python'
CASES = [('clean', 'click_bell', 0), ('randomized', 'click_bell', 0),
         ('randomized', 'click_bell', 19)]
PORT = 29881


def put(path, data):
    with path.open('x') as f:
        json.dump(data, f, indent=2)


def digest(value):
    import numpy as np
    if hasattr(value, 'detach'):
        import torch
        a = value.detach().contiguous().cpu()
        return hashlib.sha256(str((tuple(a.shape), str(a.dtype))).encode()+a.view(torch.uint8).numpy().tobytes()).hexdigest()
    if isinstance(value, np.ndarray):
        return hashlib.sha256(str((value.shape, str(value.dtype))).encode()+value.tobytes()).hexdigest()
    if isinstance(value, dict):
        return {k: digest(v) for k, v in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [digest(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def server(root):
    import random
    import numpy as np
    import torch
    import wan_va_server as module
    from configs import VA_CONFIGS
    old = load('baseline_lingbot_server', OLD/REL/'wan_va_server.py')
    from modules.model import WanAttention
    old_model = load('baseline_lingbot_model', OLD/REL/'modules/model.py')
    old_distributed = load('baseline_lingbot_distributed', OLD/REL/'distributed/util.py')
    cache_methods = {n: getattr(WanAttention, n) for n in ['init_kv_cache', 'allocate_slots']}
    original_init = module.VA_Server.__init__
    new_methods = {n: getattr(module.VA_Server, n) for n in ['_infer', '_compute_kv_cache', '_reset', 'infer']}
    states = {}
    profile = {}
    encoded = []

    def timed(name, fn):
        def call(*args, **kwargs):
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            row = profile.setdefault(name, [0, 0.])
            row[0] += 1
            row[1] += time.perf_counter()-start
            return result
        return call

    save = timed('debug_save', module.save_async)
    module.save_async = old.save_async = save
    torch.cuda.empty_cache = timed('empty_cache', torch.cuda.empty_cache)

    def init(self, cfg):
        original_init(self, cfg)
        self.candidate_transformer = self.transformer
        cpu, gpu = torch.get_rng_state(), torch.cuda.get_rng_state()
        self.baseline_transformer = old_distributed._configure_model(
            module.load_transformer(os.path.join(cfg.wan22_pretrained_model_name_or_path, 'transformer'),
                                    torch_dtype=self.dtype, torch_device=self.device, attn_mode='torch'),
            module.shard_model, self.dtype, self.device, eval_mode=True)
        a, b = dict(self.baseline_transformer.named_parameters()), dict(self.candidate_transformer.named_parameters())
        assert a.keys() == b.keys()
        for name in a:
            left = a[name].to_local() if hasattr(a[name], 'to_local') else a[name]
            assert torch.equal(left.to(dtype=b[name].dtype), b[name]), name
        torch.set_rng_state(cpu)
        torch.cuda.set_rng_state(gpu)
        put(root/'model-weight-check.json', dict(passed=True, parameters=len(a)))
        encode = self._encode_obs

        def encode_traced(obs):
            start = time.perf_counter()
            value = encode(obs)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            encoded.append(dict(value=digest(value), seconds=elapsed))
            return value
        self._encode_obs = encode_traced

    def infer(self, obs):
        active = json.loads((root/'active.json').read_text())
        variant, key = active['variant'], active['key']
        for name, candidate_method in cache_methods.items():
            setattr(WanAttention, name, candidate_method if variant == 'candidate' else getattr(old_model.WanAttention, name))
        self.transformer = self.candidate_transformer if variant == 'candidate' else self.baseline_transformer
        if obs.get('reset'):
            if key not in states:
                assert variant == 'baseline'
                states[key] = (torch.get_rng_state(), torch.cuda.get_rng_state(), random.getstate(), np.random.get_state())
                torch.save(states[key], root/'baseline'/(key+'-rng.pt'))
            cpu, gpu, py, npstate = states[key]
            torch.set_rng_state(cpu)
            torch.cuda.set_rng_state(gpu)
            random.setstate(py)
            np.random.set_state(npstate)
        for name in ['_infer', '_compute_kv_cache', '_reset']:
            fn = new_methods[name] if variant == 'candidate' else getattr(old.VA_Server, name)
            setattr(self, name, types.MethodType(fn, self))
        impl = new_methods['infer'] if variant == 'candidate' else old.VA_Server.infer
        inputs = digest(obs)
        before = digest([torch.get_rng_state(), torch.cuda.get_rng_state()])
        profile.clear()
        encoded.clear()
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = impl(self, obs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        row = dict(inputs=inputs, output=digest(result), rng_before=before,
                   rng_after=digest([torch.get_rng_state(), torch.cuda.get_rng_state()]),
                   encoded=list(encoded), profile=dict(profile), seconds=elapsed,
                   memory_allocated=torch.cuda.memory_allocated(), memory_reserved=torch.cuda.memory_reserved())
        with (root/variant/(key+'-server.jsonl')).open('a') as f:
            f.write(json.dumps(row)+'\n')
        return result

    module.VA_Server.__init__ = init
    module.VA_Server.infer = infer
    cfg = VA_CONFIGS['robotwin']
    cfg.update(json.loads((root/'server-config.json').read_text()))
    cfg.param_dtype = torch.bfloat16
    module.init_logger()
    module.run(types.SimpleNamespace(config_name='robotwin', port=PORT, save_root=cfg.save_root))


def client(root):
    import random
    import numpy as np
    from XPolicyLab.policy.LingBot_VA.lingbot_va.evaluation.robotwin.shared_plan_policy import SharedPlanPolicy
    active = json.loads((root/'active.json').read_text())
    phase, task, ordinal = active['case']
    e = load('frozen_eval', F/'scripts/c2r/evaluate.py')
    lock = e.shared_plan_lock()
    plans = e.load_shared_plans(dict(episodes=100, cases=[c for c in lock['cases'] if c['phase'] == phase]))
    official, device = e.setup_runtime()
    cfg = json.loads((root/'server-config.json').read_text())
    policy = SharedPlanPolicy(host='127.0.0.1', port=PORT, stereo=False,
                              video_guidance_scale=cfg['guidance_scale'], action_guidance_scale=cfg['action_guidance_scale'])
    original_environment = e.environment
    observations, actions = [], []

    def environment(*args):
        env, config = original_environment(*args)
        get_obs, take = env.get_obs, env.take_action

        def observed():
            start = time.perf_counter()
            value = get_obs()
            elapsed = time.perf_counter()-start
            observations.append(dict(value=digest(value), seconds=elapsed, dynamic_light=bool(env.crazy_random_light)))
            return value

        def action(value, **kwargs):
            start = time.perf_counter()
            result = take(value, **kwargs)
            elapsed = time.perf_counter()-start
            actions.append(dict(value=np.asarray(value).tolist(), seconds=elapsed,
                                rng=digest([random.getstate(), np.random.get_state()])))
            return result
        env.get_obs, env.take_action = observed, action
        return env, config

    e.environment = environment
    case = next(c for c in lock['cases'] if c['phase'] == phase and c['task'] == task)
    episode = plans[task+'--'+phase]['episodes'][ordinal]
    result = e.rollout_case(official, device, policy, case, [episode])[0]
    put(root/active['variant']/(active['key']+'.json'), dict(result=result, observations=observations, actions=actions))
    policy.client._ws.close()


def compare(root, key, other):
    reference = 'repeat' if other == 'candidate' and (root/'repeat'/(key+'.json')).exists() else 'baseline'
    a, b = [json.loads((root/v/(key+'.json')).read_text()) for v in [reference, other]]
    for field in ['seed', 'instruction', 'success', 'actions', 'initial_scene_sha256', 'initial_observation_sha256']:
        assert a['result'][field] == b['result'][field], field
    for field in ['actions', 'observations']:
        assert [{k:v for k,v in row.items() if k!='seconds'} for row in a[field]] == [{k:v for k,v in row.items() if k!='seconds'} for row in b[field]], field
    rows = [[json.loads(l) for l in (root/v/(key+'-server.jsonl')).read_text().splitlines()] for v in [reference, other]]
    for field in ['inputs', 'output', 'rng_before', 'rng_after']:
        assert [r[field] for r in rows[0]] == [r[field] for r in rows[1]], field
    assert [[v['value'] for v in r['encoded']] for r in rows[0]] == [[v['value'] for v in r['encoded']] for r in rows[1]], 'encoded latents'
    summary = dict(passed=True, case=key, comparison=other, reference=reference, baseline=a['result'], candidate=b['result'],
                   total_observations=[len(v['observations']) for v in [a,b]], rpc_counts=[len(v) for v in rows],
                   rpc_seconds=[sum(r['seconds'] for r in v) for v in rows],
                   render_seconds=[sum(r['seconds'] for r in v['observations']) for v in [a,b]],
                   simulation_seconds=[sum(r['seconds'] for r in v['actions']) for v in [a,b]],
                   speedup=a['result']['seconds']/b['result']['seconds'])
    suffix = '-warm' if reference == 'repeat' else ''
    put(root/(key+'-'+other+suffix+'-comparison.json'), summary)
    print(json.dumps(summary), flush=True)



def check_cache(device):
    """Exercise the production cache methods, including partial eviction ties."""
    import ast
    import torch
    names = {'clear_pred_cache', 'clear_cache', 'init_kv_cache', 'allocate_slots',
             '_next_cache_id', 'update_cache', 'restore_cache'}
    objects = []
    for repo in [OLD, NEW]:
        tree = ast.parse((repo/REL/'modules/model.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WanAttention')
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
        scope = {'torch': torch}
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(repo), 'exec'), scope)
        obj = types.SimpleNamespace(attn_caches={})
        for name in names:
            setattr(obj, name, types.MethodType(scope[name], obj))
        obj.init_kv_cache('test', 37, 2, 4, device, torch.float32, 2)
        objects.append(obj)
    for step in range(80):
        size = [7, 13, 4, 19, 2, 11][step % 6]
        key = torch.arange(2*size*2*4, device=device, dtype=torch.float32).reshape(2, size, 2, 4)+step
        value = -key
        slots = [obj.update_cache('test', key, value, step % 3 == 0) for obj in objects]
        assert torch.equal(slots[0].cpu(), slots[1].cpu()), ('slots', step)
        if step % 4 == 0:
            for obj, ids in zip(objects, slots): obj.restore_cache('test', ids)
        if step % 7 == 0:
            for obj in objects: obj.clear_pred_cache('test')
        a, b = [obj.attn_caches['test'] for obj in objects]
        for field in ['id', 'mask', 'is_pred']:
            assert torch.equal(a[field].cpu(), b[field].cpu()), (field, step)
        valid = a['mask'].nonzero().squeeze(-1)
        for field in ['k', 'v']:
            assert torch.equal(a[field][:, valid].cpu(), b[field][:, valid.cpu()].cpu()), (field, step)
    return dict(passed=True, operations=80, device=str(device))


def run(root, baseline_only, long_cases=False, random_long_only=False):
    cases = [CASES[0], ('clean', 'open_microwave', 99), ('randomized', 'open_microwave', 1)] if long_cases else CASES
    if random_long_only:
        cases = [CASES[0], ('randomized', 'open_microwave', 1)]
    root.mkdir(parents=True, exist_ok=True)
    for v in ['baseline', 'repeat', 'candidate', 'server']:
        (root/v).mkdir()
    gpu = subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,pci.device_id,compute_cap,memory.total','--format=csv,noheader'],text=True)
    assert len(gpu.strip().splitlines()) == 1 and '0x26BA10DE, 9.0' in gpu, gpu
    sources = [(OLD,'815a2d3b3b0a9fa8cf2bf1475257c5c323bb94b1'), (F,'a8a62f82598e0cc3c47426451fd4fda690c3198a'),
               (P/'XPolicyLab','815a2d3b3b0a9fa8cf2bf1475257c5c323bb94b1')]
    for repo, sha in sources:
        assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip() == sha
        assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
    assert not subprocess.check_output(['git','-C',str(NEW),'status','--porcelain'],text=True).strip()
    put(root/'cache-check.json', check_cache('cuda'))
    assert json.loads((CK/'complete.json').read_text())['step'] == 50000
    cfg = json.loads((T/'lingbot-spec.json').read_text())['overrides']
    bundle = root/'model-bundle'
    bundle.mkdir()
    base = Path(cfg['wan22_pretrained_model_name_or_path'])
    for name in ['vae','tokenizer','text_encoder','transformer']:
        target = CK/name if name == 'transformer' else base/name
        assert target.exists(), target
        (bundle/name).symlink_to(target, target_is_directory=True)
    cfg.update(wan22_pretrained_model_name_or_path=str(bundle), infer_mode='server', save_root=str(root/'server'), host='127.0.0.1', port=PORT, enable_wandb=False)
    put(root/'server-config.json',cfg)
    put(root/'contract.json',dict(gpu=gpu, checkpoint=str(CK), cases=cases, baseline_only=baseline_only,
                                 candidate_sha=subprocess.check_output(['git','-C',str(NEW),'rev-parse','HEAD'],text=True).strip()))
    libs = json.loads((B/'runs/fastwam-h800-smoke-20261008/system-libs-prepared.json').read_text())
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               CUDA_VISIBLE_DEVICES='0', WANDB_MODE='disabled', LD_LIBRARY_PATH=libs['ld_library_path'], VK_ICD_FILENAMES=libs['vulkan_icd'],
               PATH=libs['prefix']+'/usr/bin:/usr/local/cuda/bin:'+os.environ['PATH'],
               __EGL_VENDOR_LIBRARY_FILENAMES=str(B/'envs/system-libs/nvidia-egl-vendor.json'))
    senv = dict(env, PYTHONPATH=str(NEW/REL)+':'+str(P), MASTER_ADDR='127.0.0.1', MASTER_PORT='29882', RANK='0', LOCAL_RANK='0', WORLD_SIZE='1')
    cenv = dict(env, PYTHONPATH=':'.join(map(str,[B/'runs/shared-robotwin-acceptance-20261021/client-deps',F/'src',P/'OpenWAM',P])))
    server_process = None
    signal.signal(signal.SIGTERM, lambda *args: sys.exit(143))
    try:
        with (root/'server.log').open('x') as log:
            server_process = subprocess.Popen([str(SPY),'-u',__file__,'server',str(root)],cwd=NEW,env=senv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        deadline = time.monotonic()+1800
        while 'server listening' not in (root/'server.log').read_text(errors='replace'):
            if server_process.poll() is not None: raise RuntimeError((root/'server.log').read_text()[-5000:])
            if time.monotonic()>deadline: raise TimeoutError('server startup')
            time.sleep(3)
        print('SERVER_READY',flush=True)
        for index, case in enumerate(cases):
            key = '-'.join(map(str,case))
            variants = ['baseline', 'repeat'] if index == 0 else ['baseline']
            if not baseline_only: variants.append('candidate')
            for variant in variants:
                (root/'active.json').write_text(json.dumps(dict(variant=variant,key=key,case=case)))
                with (root/variant/(key+'.log')).open('x') as log:
                    subprocess.run([str(CPY),'-u',__file__,'client',str(root)],cwd=F,env=cenv,stdout=log,stderr=subprocess.STDOUT,check=True)
                if variant != 'baseline': compare(root,key,variant)
                print('COMPLETE',key,variant,flush=True)
    finally:
        if server_process is not None and server_process.poll() is None:
            os.killpg(server_process.pid,signal.SIGTERM)
            try: server_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server_process.pid,signal.SIGKILL)
                server_process.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=['run','server','client'])
    parser.add_argument('root',type=Path)
    parser.add_argument('--baseline-only',action='store_true')
    parser.add_argument('--long-cases',action='store_true')
    parser.add_argument('--random-long-only',action='store_true')
    args = parser.parse_args()
    if args.mode == 'run': run(args.root,args.baseline_only,args.long_cases,args.random_long_only)
    else: globals()[args.mode](args.root)
