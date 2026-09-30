"""Round-5 resource budget and paced GPU work; fail closed when monitoring fails."""
import json
import math
import subprocess
import time

import psutil
import torch
from run_pilot import loss_for

CAP_GIB=4.0
MAX_TOTAL_MIB=8192
MAX_TEMP=78
PAUSE=.04
telemetry=[]
last_check=0.


def check(force=False):
    global last_check
    now=time.monotonic()
    if not force and now-last_check<2:return
    result=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.free,temperature.gpu',
                           '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,check=True)
    used,free,temp=map(float,result.stdout.strip().splitlines()[0].split(','))
    row={'time':time.time(),'used_mib':used,'free_mib':free,'temperature_c':temp,
         'ram_available_gib':psutil.virtual_memory().available/1024**3}
    telemetry.append(row);last_check=now
    if used>=MAX_TOTAL_MIB or temp>=MAX_TEMP or row['ram_available_gib']<4:
        raise RuntimeError('Resource guard stopped work: '+json.dumps(row))


def initialize():
    check(True)
    if telemetry[-1]['free_mib']<5*1024:
        raise RuntimeError('Need at least 5 GiB free VRAM before starting')
    torch.cuda.set_per_process_memory_fraction(CAP_GIB*1024**3/torch.cuda.get_device_properties(0).total_memory)


def pause():
    torch.cuda.synchronize()
    time.sleep(PAUSE)
    check()


@torch.no_grad()
def evaluate(model,batches,device):
    losses=[]
    for ids in batches:
        check()
        losses.append(float(loss_for(model,ids.to(device))))
        pause()
    if not all(math.isfinite(v) for v in losses):raise RuntimeError('Nonfinite CE')
    ce=sum(losses)/len(losses)
    return {'ce':ce,'ppl':math.exp(ce),'sequence_ce':losses,
            'scored_tokens':len(batches)*(batches[0].shape[1]-1)}


def capture(model,module,batches,device,gradients):
    xs,zs,gs=[],[],[]
    for ids in batches:
        check();cache={}
        def hook(_module,inputs,output):
            cache['x']=inputs[0].detach().float().cpu().reshape(-1,inputs[0].shape[-1]).T
            cache['z']=output.detach().requires_grad_(True) if gradients else output.detach()
            return cache['z']
        handle=module.register_forward_hook(hook)
        try:
            with torch.set_grad_enabled(gradients):
                loss=loss_for(model,ids.to(device))
                if gradients:
                    g,=torch.autograd.grad(loss,cache['z'])
                    gs.append(g.detach().float().cpu().reshape(-1,g.shape[-1]).T)
            xs.append(cache['x'])
            z=cache['z'].detach().float().cpu();zs.append(z.reshape(-1,z.shape[-1]).T)
        finally:handle.remove()
        pause()
    return torch.cat(xs,1),torch.cat(zs,1),torch.cat(gs,1) if gradients else None


def summary():
    return {'allocator_cap_gib':CAP_GIB,'global_memory_stop_mib':MAX_TOTAL_MIB,'temperature_stop_c':MAX_TEMP,
            'pause_seconds':PAUSE,'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
            'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,'samples':telemetry}
