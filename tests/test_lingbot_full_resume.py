"""Exercise production checkpoint methods with AdamW, worker RNG and data replay."""
import ast
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.distributed.checkpoint.state_dict import get_model_state_dict, get_optimizer_state_dict, set_optimizer_state_dict, set_model_state_dict, StateDictOptions
from safetensors.torch import save_file, load_file
import logging
logger = logging.getLogger(__name__)

class Samples(Dataset):
    def __len__(self): return 17
    def __getitem__(self, index):
        return torch.tensor([index / 17, random.random(), float(np.random.rand()), float(torch.rand(()))])

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.Dropout(.2), torch.nn.Linear(16, 1))
        self.later = torch.nn.Linear(4, 1)
        self.unused = torch.nn.Linear(4, 1)

    def forward(self, x, activate_later):
        y = self.main(x)
        return y + self.later(x) if activate_later else y

def main():
    source = Path(__file__).resolve().parents[1] / 'policy/LingBot_VA/lingbot_va/wan_va/train.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Trainer')
    names = {'_rng_state', '_restore_rng', '_start_data_epoch', '_get_next_batch', '_resume_contract', 'save_checkpoint', '_load_training_state'}
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), 'exec'), globals())
    out = Path(os.environ['RESUME_TEST_OUTPUT']); out.mkdir(parents=True, exist_ok=True)
    rank=int(os.environ.get('RANK',0)); world=int(os.environ.get('WORLD_SIZE',1)); local=int(os.environ.get('LOCAL_RANK',0))
    torch.cuda.set_device(local)
    if 'RANK' in os.environ: dist.init_process_group('nccl')
    else: dist.init_process_group('nccl', init_method='tcp://127.0.0.1:29687', rank=0, world_size=1)
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    def make():
        t = Trainer()
        t.config = SimpleNamespace(world_size=world, rank=rank, local_rank=local, batch_size=2, load_worker=2)
        t.device = torch.device(f'cuda:{local}'); t.step = 0
        t.transformer = Model().cuda()
        t.transformer.config = {}
        fully_shard(t.transformer, mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32))
        t.optimizer = torch.optim.AdamW(t.transformer.parameters(), lr=.001, fused=True)
        t.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(t.optimizer, lambda step: min((step+1)/5, 1))
        t.gradient_accumulation_steps = 2
        dataset=Samples(); sampler=DistributedSampler(dataset, num_replicas=world, rank=rank, seed=42)
        t.train_loader = DataLoader(dataset, batch_size=2, sampler=sampler, num_workers=2, generator=torch.Generator().manual_seed(42+rank))
        t.train_loader_iter = None; t.data_epoch = 0; t.data_batches = 0; t.epoch_rng = None
        t.save_dir = out
        return t
    def update(t):
        batches=[]; losses=[]
        for _ in range(2):
            x=t._get_next_batch().cuda(); batches.append(x.cpu())
            loss=(t.transformer(x, t.step >= 3).float()-torch.randn(x.shape[0],1,device='cuda')).square().mean()/2
            loss.backward(); losses.append(loss.detach().cpu())
        t.optimizer.step(); t.lr_scheduler.step(); t.optimizer.zero_grad(); t.step+=1
        return batches, losses
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    a=make()
    for _ in range(2): update(a)
    a.save_checkpoint()
    expected=[update(a) for _ in range(4)]
    opts=StateDictOptions(full_state_dict=True, cpu_offload=True)
    weights=get_model_state_dict(a.transformer, options=opts)
    optimizer=get_optimizer_state_dict(a.transformer,a.optimizer,options=opts)
    scheduler=a.lr_scheduler.state_dict()
    b=make()
    from torch.distributed.checkpoint.state_dict import set_model_state_dict
    set_model_state_dict(b.transformer,load_file(out/'checkpoint_step_2/transformer/diffusion_pytorch_model.safetensors'),options=opts)
    b._load_training_state(out/'checkpoint_step_2')
    assert all(p.requires_grad for p in b.transformer.parameters())
    assert all(p not in b.optimizer.state for n,p in b.transformer.named_parameters() if n.startswith(('later.', 'unused.')))
    actual=[update(b) for _ in range(4)]
    def equal(x,y):
        if isinstance(x,torch.Tensor): assert torch.equal(x,y), (x,y)
        elif isinstance(x,dict):
            assert x.keys()==y.keys()
            for k in x: equal(x[k],y[k])
        elif isinstance(x,(list,tuple)):
            assert len(x)==len(y)
            for u,v in zip(x,y): equal(u,v)
        else: assert x==y,(x,y)
    equal(expected,actual); equal(weights,get_model_state_dict(b.transformer,options=opts))
    equal(optimizer,get_optimizer_state_dict(b.transformer,b.optimizer,options=opts)); equal(scheduler,b.lr_scheduler.state_dict())
    assert b.step==a.step==6 and b.data_epoch==a.data_epoch
    b.save_checkpoint()
    assert (out/'checkpoint_step_2/training_state.pt').is_file()
    assert (out/'checkpoint_step_2/complete.json').is_file()
    assert (out/'checkpoint_step_6/complete.json').is_file()
    assert all(t.dtype == torch.bfloat16 for t in load_file(out/'checkpoint_step_2/transformer/diffusion_pytorch_model.safetensors').values())
    c=make(); c._load_training_state(out/'checkpoint_step_6')
    equal(weights,get_model_state_dict(c.transformer,options=opts))
    equal(optimizer,get_optimizer_state_dict(c.transformer,c.optimizer,options=opts))
    try: b._load_training_state(out/'missing')
    except RuntimeError: pass
    else: raise AssertionError('missing checkpoint accepted')
    result=dict(passed=True,fsdp2=True,world_size=world,optimizer_steps=6,save_step=2,resumed_steps=4,worker_count=2,gradient_accumulation=2,exact=['batches','losses','fp32_weights','adamw_state','scheduler','step','epoch'],scope='small model using production Trainer checkpoint methods')
    if rank==0:
        (out/'result.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result))
    dist.destroy_process_group()
if __name__=='__main__': main()
