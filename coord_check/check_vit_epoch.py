import math, os, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm
from copy import copy 

from fastDP import PrivacyEngine
from opacus.validators import ModuleValidator 
from .plot_coord_data import plot_coord_data

import numpy as np
import argparse

from scripts.get_params import get_shapes, _get_noise4target, _get_lr4target, _get_clip4target
from scripts.model_builder import MyVit, MyPreVit

import warnings; 
warnings.filterwarnings("ignore")



args = argparse.Namespace(
    model="vit_tiny_patch16_224",
    noise=0.9,
    lr=5, epochs=3, bs=100, mini_bs=100,
    dimension=224,
    dataset='CIFAR10',
    clipping_mode='BK-MixOpt',
    clipping_style='layer-wise',
    origin_params=None,
    device='cuda',
    bptt=1000,
    precision='float32'
)

  
device = torch.device("cuda:0")


def my_custom_optimizer_fn(net, args, trainset_len, mode='full'):
    # 获取全局 device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Base Model (计算 Shape 用，不需要上 GPU)
    model_base = MyVit(args, is_base=True)
    base_model = model_base.create_model()
    base_shapes = get_shapes(base_model)
    
    # 2. Target Model (使用传入的 net)
    model_shapes = get_shapes(net)

    # 3. 计算超参
    noise = _get_noise4target(base_shapes, model_shapes, base_noise=args.noise)
    clip_dict = _get_clip4target(base_shapes, model_shapes, target_noise=noise)
    
    # ★关键修改：必须把 clipping vector 放到 GPU 上
    D_prime_vector = torch.stack(list(clip_dict.values())).to(device)

    # 4. 修复模型层并上 GPU
    net = ModuleValidator.fix(net) 
    net = net.to(device)

    # 5. 计算 LR
    base_lr = 2 ** args.lr
    target_lr_dict = _get_lr4target(base_shapes, model_shapes, args.noise, noise, base_lr)

    param_groups = []
    for n, p in net.named_parameters():
        curr_lr = target_lr_dict.get(n, base_lr)
        if isinstance(curr_lr, torch.Tensor):
            curr_lr = curr_lr.item()
            
        param_groups.append({
            "params": [p], 
            "lr": curr_lr, 
            "name": n
        })
    
    # optimizer = optim.SGD(param_groups, lr=base_lr)
    optimizer = optim.SGD(net.parameters(), lr=base_lr)

    # 6. Privacy Engine
    if 'nonDP' not in args.clipping_mode:
        if 'BK' in args.clipping_mode:
            clipping_mode = args.clipping_mode[3:]
        else:
            clipping_mode = 'ghost'

        if isinstance(args.clipping_style, list):
            args.clipping_style = args.clipping_style[0]

        privacy_engine = PrivacyEngine(
            net,
            batch_size=args.bs,
            sample_size=trainset_len,
            noise_multiplier=args.noise,
            epochs=args.epochs,
            clipping_mode=clipping_mode,
            # clipping_coe=D_prime_vector, 
            clipping_style=args.clipping_style,
            origin_params=args.origin_params,
        )
        privacy_engine.attach(optimizer)
        # 打印一下确认 Noise 算出来了
        # print(f"Noise multiplier (σ): {privacy_engine.noise_multiplier:.4f}")
        
    return optimizer


def _record_coords(df, width, name, batch_idx, output_fdict=None, input_fdict=None, param_fdict=None):
    def hook(module, inputs, outputs):
        record = {
            'width': width,
            'module': name,
            'batch_idx': batch_idx,
            't': batch_idx,
        }
        if outputs is not None and isinstance(outputs, torch.Tensor):
            record['activation_std'] = outputs.std().item()
            record['activation_mean'] = outputs.mean().item()
            record['l1'] = outputs.abs().mean().item()

        if input_fdict is not None:
            x_in = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
            if isinstance(x_in, torch.Tensor):
                input_fdict.setdefault(width, []).append(x_in.std().item())

        if param_fdict is not None:
            for pname, p in module.named_parameters(recurse=False):
                if p.grad is not None:
                    key = f"{name}.{pname}"
                    param_fdict.setdefault(key, []).append(p.grad.std().item())

        df.append(record)
    return hook


def _get_coord_data(models,
                    dataloader=None,
                    optimizer_fn=None,
                    nsteps=3,       # 如果启用 epochs，这个参数将被忽略或仅用于 fix_data
                    epochs=None,    # ★ 新增：传入 args.epochs
                    flatten_input=False,
                    flatten_output=False,
                    lossfn='xent',
                    fix_data=False, # 跑 Epoch 通常建议 False (遍历整个数据集)
                    cuda=True,
                    nseeds=1,
                    show_progress=True
                    ):
    import pandas as pd
    coord_data_list = []

    # 1. 检查 CUDA
    if cuda and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        cuda = False
    
    device = torch.device("cuda" if cuda else "cpu")


    if fix_data and not isinstance(dataloader, list):
        # 如果是 fix_data，这里依然只是取出一个 batch 重复
        # 但为了模拟 "Epoch"，长度可能需要设置一下，或者干脆由 dataloader 本身长度决定
        batch = next(iter(dataloader))
        # 这里的 nsteps 变成了 "一个 Epoch 有多少个 Batch"
        dataloader = [batch] * (nsteps if nsteps else 10) 

    # 获取每个 Epoch 的总 Batch 数，用于判断何时是 "最后一个 Batch"
    total_batches = len(dataloader)

    if show_progress:
        from tqdm import tqdm
        # 进度条总长 = 种子数 * 模型数 * Epoch 数
        total_steps = nseeds * len(models) * (epochs if epochs else 1)
        pbar = tqdm(total=total_steps)

    for i in range(nseeds):
        torch.manual_seed(i)
        for width, model_ctor in models.items():
            model = model_ctor()
            model.train()
            
            if cuda:
                model = model.to(device)

            optimizer = optimizer_fn(model)

            # === 外层循环：Epochs ===
            # 如果没传 epochs，默认为 1
            run_epochs = epochs if epochs is not None else 1
            
            for epoch in range(1, run_epochs + 1):
                
                # 内层循环：Batches
                for batch_idx, batch in enumerate(dataloader, 1):
                    
                    # ★ 关键修改：只在当前 Epoch 的最后一个 Batch 记录 ★
                    # 这样横坐标 t 就是 epoch，而不是 step
                    is_record_step = (batch_idx == total_batches)
                    
                    remove_hooks = []
                    
                    if is_record_step:
                        # 只有需要记录时，才注册 Hook
                        for name, module in model.named_modules():
                            weight = getattr(module, 'weight', None)
                            if weight is None or not isinstance(weight, torch.Tensor):
                                continue
                            
                            # 筛选 Linear(2D) 和 Conv(4D)
                            if weight.ndim in [2, 4]:
                                # 注意：这里的 batch_idx 参数我们传入 epoch
                                # 这样画图时 x 轴显示的就是 1, 2, 3 (Epoch)
                                remove_hooks.append(module.register_forward_hook(
                                    _record_coords(coord_data_list, width, name, epoch)))

                    (data, target) = batch
                    
                    if cuda:
                        data = data.to(device, non_blocking=True)
                        target = target.to(device, non_blocking=True)
                    
                    if flatten_input:
                        data = data.view(data.size(0), -1)

                    output = model(data)
                    
                    if flatten_output:
                        output = output.view(-1, output.shape[-1])

                    if lossfn == 'xent':
                        loss = F.cross_entropy(output, target)
                    else:
                        raise NotImplementedError

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # 移除 Hook
                    for handle in remove_hooks:
                        handle.remove()

                # 每个 Epoch 结束后更新进度条
                if show_progress:
                    pbar.update(1)

    if show_progress:
        pbar.close()

    return pd.DataFrame(coord_data_list)


def get_coord_data(models, dataloader, optimizer_fn, **kwargs):
    df = _get_coord_data(models, dataloader, optimizer_fn, **kwargs)
    df['optimizer'] = 'custom'
    return df

def setprec(model, precision='float32'):
    if precision == 'float32':
        return model.to(dtype=torch.float32)
    elif precision == 'float16':
        return model.to(dtype=torch.float16)
    elif precision == 'bfloat16':
        return model.to(dtype=torch.bfloat16)
    else:
        raise ValueError(f"Unsupported precision: {precision}")



def coord_check_split_terms(lr, model_fn, optimizer_fn, batch_size, nsteps, nseeds, args):
                    
    def gen(s):
        def f():
            local_args = copy(args)
            # === 修正 1: 这里必须用传入的参数 w ===
            local_args.scale = s  
            
            model_wrapper = MyVit(local_args, is_base=False)
            net = model_wrapper.create_model()

            def kaiming_init_weights(m):
                if isinstance(m, nn.Linear):
                    if m is net.head:
                        nn.init.zeros_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    else:
                        nn.init.kaiming_normal_(m.weight, a=1, mode='fan_in')
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    
            net.apply(kaiming_init_weights)
            
            net = setprec(net, args.precision)
            return net
        return f

    scales = np.arange(1, 7) 
    models = {int(s): gen(int(s)) for s in scales}
                      
    transformation = torchvision.transforms.Compose([
        torchvision.transforms.Resize(args.dimension),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    if args.dataset == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root='data/', train=True, download=True, transform=transformation)
    elif args.dataset == 'CIFAR100':
        trainset = torchvision.datasets.CIFAR100(root='data/', train=True, download=True, transform=transformation)
    else:
        raise ValueError("Must specify dataset as CIFAR10 or CIFAR100.")

    full_trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    for mode in ['bs', 'scale', 'both']:
        df = _get_coord_data(
            models,
            dataloader=full_trainloader,   # ★ 用 per-width 数据
            optimizer_fn=lambda net: optimizer_fn(net, args, 50000, mode=mode),
            flatten_output=True,
            nseeds=nseeds,
            epochs=args.epochs, # 告诉函数我们要跑 Epoch
            lossfn='xent',
            fix_data=False,   # 已经自己重复过 batch 了
        )
        
        plot_coord_data(
            df,
            y='l1',
            legend=False,
            loglog=False,
            save_to=f"/content/SGD_sp.pdf", 
            suptitle=None,
            face_color=None
        )

coord_check_split_terms(
    lr=args.lr,
    model_fn=None,
    optimizer_fn=my_custom_optimizer_fn,
    batch_size=args.mini_bs,
    nsteps=None, # 这里不再重要了
    nseeds=1,
    args=args
)
