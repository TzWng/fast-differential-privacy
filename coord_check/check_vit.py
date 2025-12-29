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
    lr=-4, epochs=3, bs=200, mini_bs=200,
    dimension=224,
    dataset_name='CIFAR10', cifar_data='CIFAR10',
    clipping_mode='BK-MixOpt',
    clipping_style='layer-wise',
    origin_params=None,
    device='cuda',
    bptt=1000,
    precision='float32'
)

  
device = torch.device("cuda:0")


def my_custom_optimizer_fn(net, args, trainset_len, mode='full'):

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
                  
    model_base = MyVit(args, is_base=True)
    base_model = model_base.create_model() 
    base_shapes = get_shapes(base_model)
    
    model_target = MyVit(args, is_base=False)    
    net = model_target.create_model()
    net.apply(kaiming_init_weights)
    model_shapes = get_shapes(net)

    
    noise = _get_noise4target(base_shapes, model_shapes, base_noise=args.noise)
    clip_dict = _get_clip4target(base_shapes, model_shapes, target_noise=noise)
    D_prime_vector = torch.stack(list(clip_dict.values()))

    net = ModuleValidator.fix(net)
    net = net.to(device)

    print('Number of total parameters: ', sum([p.numel() for p in net.parameters()]))
    print('Number of trainable parameters: ', sum([p.numel() for p in net.parameters() if p.requires_grad]))
  
    criterion = F.cross_entropy

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

    # optimizer = torch.optim.SGD(param_groups, lr=2 ** args.lr) # muP
    # optimizer = torch.optim.SGD(net.parameters(), lr=2 ** args.lr) # SP
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
            clipping_coe=D_prime_vector,
            clipping_style=args.clipping_style,
            origin_params=args.origin_params,
        )
        privacy_engine.attach(optimizer)
        print(f"Noise multiplier (σ): {privacy_engine.noise_multiplier:.4f}")

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
                    cuda=True,
                    nseeds=1,
                    show_progress=True
                    ):
    import pandas as pd
    coord_data_list = []

    # === 核心修改：移除了 dataloader_map 判断逻辑 ===
    # 恢复原有的 fix_data 逻辑：如果需要固定数据，就取出一个 batch 重复 nsteps 次
    if fix_data:
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
            if cuda:
                model = model.cuda()
            optimizer = optimizer_fn(model)

            # === 直接使用统一的 dataloader ===
            for batch_idx, batch in enumerate(dataloader, 1):
                remove_hooks = []

                for name, module in model.named_modules():
                    remove_hooks.append(module.register_forward_hook(
                        _record_coords(coord_data_list, width, name, batch_idx)))

                (data, target) = batch
                if cuda:
                    data, target = data.cuda(), target.cuda()
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
    def gen(w):
        def f():
            local_args = copy(args)
            # === 修正 1: 这里必须用传入的参数 w ===
            local_args.width = w  
            
            model_wrapper = MyVit(local_args, is_base=False)
            model = model_wrapper.create_model()
            
            model = setprec(model, args.precision)
            return model
        return f

    scales = 192 * np.arange(1, 7) 
    models = {int(s): gen(int(s)) for s in scales}
                      
    transformation = torchvision.transforms.Compose([
        torchvision.transforms.Resize(args.dimension),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    if args.cifar_data == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root='data/', train=True, download=True, transform=transformation)
    elif args.cifar_data == 'CIFAR100':
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
            loglog=True,
            save_to=f"/content/SGD_mup.pdf", 
            suptitle=None,
            face_color=None
        )

coord_check_split_terms(
    lr=args.lr,
    model_fn=None,
    optimizer_fn=my_custom_optimizer_fn,
    batch_size=args.mini_bs,
    nsteps=4,
    nseeds=3,
    args=args
)
