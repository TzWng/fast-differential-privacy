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
from .plot_coord_data import plot_coord_data

import numpy as np
import argparse

from scripts.get_params import get_shapes, _get_noise4target, _get_lr4target, _get_clip4target
from scripts.model_builder import MyVit, MyPreVit

import warnings; 
warnings.filterwarnings("ignore")



args = argparse.Namespace(
    model="vit_tiny_patch16_224",
    lr=-4, epochs=3, bs=500, mini_bs=500,
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

    # 1. Base Model (计算 Shape 用，不需要上 GPU)
    model_base = MyVit(args, is_base=True)
    base_model = model_base.create_model()
    base_shapes = get_shapes(base_model)
    
    # 2. Target Model (使用传入的 net)
    net.apply(kaiming_init_weights)
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
    
    optimizer = optim.SGD(param_groups, lr=base_lr)

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
            noise_multiplier=noise,
            epochs=args.epochs,
            clipping_mode=clipping_mode,
            clipping_coe=D_prime_vector, # 现在是 GPU Tensor 了
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
                    nsteps=3,
                    flatten_input=False,
                    flatten_output=False,
                    lossfn='xent',
                    fix_data=True, 
                    cuda=True,     # 确保这里默认为 True
                    nseeds=1,
                    show_progress=True
                    ):
    import pandas as pd
    coord_data_list = []

    # 1. 检查环境
    if cuda and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        cuda = False
    
    device = torch.device("cuda" if cuda else "cpu")

    # 2. 处理数据固定 (List or Loader)
    if fix_data and not isinstance(dataloader, list):
        batch = next(iter(dataloader))
        dataloader = [batch] * nsteps

    if show_progress:
        from tqdm import tqdm
        pbar = tqdm(total=nseeds * len(models))

    for i in range(nseeds):
        torch.manual_seed(i)
        for width, model_ctor in models.items():
            model = model_ctor()
            model.train()
            
            # ★ 3. 模型强制上 GPU
            if cuda:
                model = model.to(device)
            
            # [Debug] 打印一次模型位置
            if i == 0 and list(models.keys())[0] == width:
                first_param = next(model.parameters())
                print(f"\n[Check] Model (width={width}) device: {first_param.device}")

            optimizer = optimizer_fn(model)

            for batch_idx, batch in enumerate(dataloader, 1):
                remove_hooks = []
                # 注册 Hook
                for name, module in model.named_modules():
                    remove_hooks.append(module.register_forward_hook(
                        _record_coords(coord_data_list, width, name, batch_idx)))

                (data, target) = batch
                
                # ★ 4. 数据强制上 GPU
                if cuda:
                    data = data.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                
                # [Debug] 打印一次数据位置
                if i == 0 and batch_idx == 1 and list(models.keys())[0] == width:
                     print(f"[Check] Input Data device: {data.device}")

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

                for handle in remove_hooks:
                    handle.remove()

                if batch_idx == nsteps:
                    break

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
            model = model_wrapper.create_model()
            
            model = setprec(model, args.precision)
            return model
        return f

    scales = np.arange(1, 3) 
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

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    batch = next(iter(trainloader))
    fixed_dataloader = [batch] * nsteps

    for mode in ['bs', 'scale', 'both']:
        df = _get_coord_data(
            models,
            dataloader=fixed_dataloader,   # ★ 用 per-width 数据
            optimizer_fn=lambda net: optimizer_fn(net, args, 50000, mode=mode),
            flatten_output=True,
            nseeds=nseeds,
            nsteps=nsteps,
            lossfn='xent',
            fix_data=False,   # 已经自己重复过 batch 了
        )
        
        plot_coord_data(
            df,
            y='l1',
            legend=True,
            loglog=False,
            save_to=f"/content/SGD_mup.pdf", 
            suptitle=None,
            face_color=None
        )

coord_check_split_terms(
    lr=args.lr,
    model_fn=None,
    optimizer_fn=my_custom_optimizer_fn,
    batch_size=args.mini_bs,
    nsteps=2,
    nseeds=1,
    args=args
)
