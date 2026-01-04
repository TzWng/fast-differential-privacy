"""
Simplified training script with muP and FastDP.
Strategy: Linear Warmup -> Fixed Learning Rate (No Decay).
"""

import os
import sys
import time
import math
import pickle
import argparse
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Custom modules
from modelDP import GPTConfig, GPT
from fastDP import PrivacyEngine as PrivacyEngine
from scripts.get_params_gpt2 import get_shapes, _get_noise4target, _get_clip4target, _get_lr4target_adam

import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 1. Argument Parsing
# -----------------------------------------------------------------------------
sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser(description='GPT-2 muP Training (Fixed LR)')

# Model Config
parser.add_argument('--n_layer', type=int, default=10, help='Number of layers')
parser.add_argument('--n_head', type=int, default=20, help='Number of heads')
parser.add_argument('--n_head_base', type=int, default=5, help='Base model heads for muP')
parser.add_argument('--block_size', type=int, default=1024, help='Context size')

# Optimization Config
parser.add_argument('--batch_size', type=int, default=16, help='Micro batch size')
parser.add_argument('--grad_accum', type=int, default=8, help='Gradient accumulation steps')
parser.add_argument('--total_steps', type=int, default=1000, help='Total training steps')
parser.add_argument('--optim_type', type=str, default='adam', choices=['adam', 'sgd'])

# muP Base Hyperparams
parser.add_argument('--base_optimal_lr', type=float, default=5e-3, help='Optimal LR for Base Model')
parser.add_argument('--base_optimal_noise', type=float, default=1.0, help='Optimal Noise for Base Model')

# DP Config
parser.add_argument('--per_sample_clip', action='store_true', default=True, help='Enable DP clipping')

# System/IO
parser.add_argument('--out_dir', type=str, default='out', help='Output directory')
parser.add_argument('--init_from', type=str, default='scratch', choices=['scratch', 'resume'])
parser.add_argument('--wandb', action='store_true', help='Enable WandB')
parser.add_argument('--compile', action='store_true', help='Use torch.compile')

args = parser.parse_args()

# ⚠️ 关键修复：configurator.py 已经被删除了，不要再 exec 它是引发 AssertionError 的元凶

# Map args to globals
out_dir = args.out_dir
eval_iters = 200
always_save_checkpoint = False
init_from = args.init_from

wandb_log = args.wandb
wandb_project = 'DPscaling'
wandb_run_name = f'gpt2-muP-H{args.n_head}-FixedLR'

dataset = 'shakespeare_char'
gradient_accumulation_steps = args.grad_accum
batch_size = args.batch_size
block_size = args.block_size

n_layer = args.n_layer
n_head = args.n_head
n_head_base = args.n_head_base
dropout = 0.0
bias = False

learning_rate = args.base_optimal_lr 
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95

backend = 'nccl'
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = args.compile

# muP Containers
noise = None
D_prime_vector = None
target_lrs = None

# -----------------------------------------------------------------------------
# 2. DDP Setup
# -----------------------------------------------------------------------------
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    num_GPUs = torch.cuda.device_count()

total_bs = gradient_accumulation_steps * batch_size * ddp_world_size
tokens_per_iter = total_bs * block_size
if master_process:
    print(f"🚀 Total Batch Size: {total_bs}, Tokens/Iter: {tokens_per_iter:,}", flush=True)
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(enabled=True, device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# 3. Data Loader
# -----------------------------------------------------------------------------
data_dir = os.path.join('data', dataset)
def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']

# -----------------------------------------------------------------------------
# 4. Model & muP Calculation
# -----------------------------------------------------------------------------
n_embd = n_head * 64 
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)
model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304

def calculate_mup_params():
    print(f"--- [muP] Calculating params Base(H={n_head_base}) -> Target(H={n_head}) ---", flush=True)
    
    # 1. Base Model
    base_args = model_args.copy()
    base_args['n_head'] = n_head_base
    base_args['n_embd'] = n_head_base * 64
    base_model = GPT(GPTConfig(**base_args))
    base_shapes = get_shapes(base_model)
    del base_model
    
    # 2. Target Model
    target_conf = GPTConfig(**model_args)
    target_model = GPT(target_conf)
    target_shapes = get_shapes(target_model)
    
    # 3. Calculate
    _noise = _get_noise4target(base_shapes, target_shapes, base_noise=args.base_optimal_noise)
    _clip_dict = _get_clip4target(base_shapes, target_shapes, target_noise=_noise)
    _D_vec = torch.stack(list(_clip_dict.values()))
    
    _target_lrs = _get_lr4target_adam(
        base_shapes, target_shapes, 
        base_noise=args.base_optimal_noise, 
        target_noise=_noise, 
        base_lr=args.base_optimal_lr
    )
    return target_model, _noise, _D_vec, _target_lrs

if init_from == 'scratch':
    print("Initializing from scratch", flush=True)
    model, noise, D_prime_vector, target_lrs = calculate_mup_params()
    iter_num = 0
    
elif init_from == 'resume':
    print(f"Resuming from {out_dir}", flush=True)
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    
    _, noise, D_prime_vector, target_lrs = calculate_mup_params()
    iter_num = checkpoint['iter_num']

model.to(device)

if master_process:
    print(f"muP: Noise={noise:.4f} | LR Scaling applied.", flush=True)

# -----------------------------------------------------------------------------
# 5. Optimizer (Layer-wise LR)
# -----------------------------------------------------------------------------
param_groups = []
for name, param in model.named_parameters():
    if not param.requires_grad: continue
    
    if name in target_lrs:
        local_lr = target_lrs[name]
    elif name.endswith('.bias'):
        weight_name = name.replace('.bias', '.weight')
        local_lr = target_lrs.get(weight_name, learning_rate)
    else:
        local_lr = learning_rate
        
    wd = weight_decay if param.dim() >= 2 else 0.0
    param_groups.append({'params': [param], 'lr': local_lr, 'weight_decay': wd, 'base_mup_lr': local_lr})

if args.optim_type == 'adam':
    optimizer = torch.optim.AdamW(param_groups, betas=(beta1, beta2))
elif args.optim_type == 'sgd':
    optimizer = torch.optim.SGD(param_groups, momentum=beta1)

if init_from == 'resume' and 'optimizer' in checkpoint:
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None 

# -----------------------------------------------------------------------------
# 6. Privacy Engine
# -----------------------------------------------------------------------------
enable_DP = args.per_sample_clip or (noise > 0)

if enable_DP:
    len_data = len(np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r'))
    privacy_engine = PrivacyEngine(
        model,
        batch_size=total_bs,
        num_steps=args.total_steps,
        sample_size=len_data,
        noise_multiplier=noise,
        num_GPUs=ddp_world_size,
        torch_seed_is_fixed=False,
        grad_accum_steps=gradient_accumulation_steps,
        clipping_mode="ghost",
        clipping_coe=D_prime_vector,
        clipping_style='layer-wise',
    )
    if master_process:
        print("=======", "Privacy Engine Loaded", "=======", flush=True)

# -----------------------------------------------------------------------------
# 7. Compile & Utils
# -----------------------------------------------------------------------------
if compile:
    print("compiling...", flush=True)
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# 🟢 极简 Scheduler: Warmup -> Constant (Fix LR)
def get_lr_mult(it):
    warmup_steps = 2000
    if it < warmup_steps:
        return (it + 1) / warmup_steps
    return 1.0 # 恒定

if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=args)

# -----------------------------------------------------------------------------
# 8. Training Loop
# -----------------------------------------------------------------------------
X, Y = get_batch('train')
t0 = time.time()
iter_num = 0 if init_from == 'scratch' else iter_num
local_iter_num = 0
running_mfu = -1.0
raw_model = model.module if ddp else model

while iter_num <= args.total_steps:
    # Update LR (Fixed)
    lr_mult = get_lr_mult(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['base_mup_lr'] * lr_mult

    # Eval
    if iter_num % 100 == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}", flush=True)
        if wandb_log:
            wandb.log({"iter": iter_num, "train/loss": losses['train'], "val/loss": losses['val']}, step=iter_num)

    # Forward Backward
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        
        with ctx:
            logits, loss = model(X, Y)
            # loss = loss / gradient_accumulation_steps (已移除)
            
        X, Y = get_batch('train')
        loss.backward()

    # DP Manifold
    if enable_DP:
        for n, p in model.named_parameters():
            if hasattr(p, 'private_grad'):
                if ddp:
                    torch.distributed.all_reduce(p.private_grad.contiguous(), op=torch.distributed.ReduceOp.SUM)
                p.grad = p.private_grad / ddp_world_size / batch_size / gradient_accumulation_steps 
                del p.private_grad

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # Logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % 10 == 0 and master_process:
        lossf = loss.item() 
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%", flush=True)

    iter_num += 1
    local_iter_num += 1

if ddp:
    destroy_process_group()
