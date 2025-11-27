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

from fastDP import PrivacyEngine
from .plot_coord_data import plot_coord_data
from opacus.accountants.utils import get_noise_multiplier  # 现在没用到，可以先留着
warnings.filterwarnings("ignore")

class MLP(nn.Module):
    def __init__(self, width=128, input_dim=3072, num_classes=10, nonlin=F.relu, output_mult=1.0, input_mult=1.0):
        super(MLP, self).__init__()
        self.nonlin = nonlin
        self.input_mult = input_mult
        self.output_mult = output_mult
        self.fc_1 = nn.Linear(input_dim, width, bias=False)
        self.fc_2 = nn.Linear(width, 2*width, bias=False)
        self.fc_3 = nn.Linear(2*width, 2*width, bias=False)
        self.fc_4 = nn.Linear(2*width, width, bias=False)
        self.fc_5 = nn.Linear(width, num_classes, bias=False) 
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.fc_1.weight, a=1, mode='fan_in')
        self.fc_1.weight.data /= self.input_mult**0.5
        nn.init.kaiming_normal_(self.fc_2.weight, a=1, mode='fan_in')
        nn.init.kaiming_normal_(self.fc_3.weight, a=1, mode='fan_in')
        nn.init.kaiming_normal_(self.fc_4.weight, a=1, mode='fan_in')
        # nn.init.kaiming_normal_(self.fc_5.weight, a=1, mode='fan_in')
        nn.init.zeros_(self.fc_5.weight)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        out = self.nonlin(self.fc_1(x) * self.input_mult**0.5)
        out = self.nonlin(self.fc_2(out))
        out = self.nonlin(self.fc_3(out))
        out = self.nonlin(self.fc_4(out))
        return self.fc_5(out) * self.output_mult

import argparse


args = argparse.Namespace(
    lr=-4, epochs=3, bs=500, mini_bs=500,
    dataset_name='CIFAR10', cifar_data='CIFAR10',

    clipping_mode='BK-MixOpt',
    clipping_style='layer-wise',
    origin_params=None,
    device='cuda',
    bptt=1000,
    precision='float32'
)

def my_mlp_fn(width):
    dim = 8 * (width / 128.0) ** 0.5 
    input_dim = 3 * dim * dim
    model = MLP(width=width, input_dim=input_dim, nonlin=torch.relu, output_mult=32, input_mult=1/256)
    return model
  
device = torch.device("cuda:0")



def my_custom_optimizer_fn(net, args, trainset_len, mode='full'):
    width = net.fc_2.weight.shape[0]
    sigma = 2.0 * (128.0/width) ** 0.5
    learning_rates = {}
    for name, param in net.named_parameters():
        size = param.shape
        adjust = (size[0] ** 0.5 + size[1] ** 0.5) * sigma / args.bs 
        learning_rates[name] = (2 ** args.lr) * (size[0] / size[1])**0.5 / adjust

    param_groups = [
        {'params': [param], 'lr': learning_rates.get(name, 2**args.lr)}
        for name, param in net.named_parameters()
    ]
    optimizer = torch.optim.SGD(param_groups, lr=2 ** args.lr) # muP
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
            noise_multiplier=sigma,
            epochs=args.epochs,
            clipping_mode=clipping_mode,
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
                    dataloader=None,          # 原来的
                    optimizer_fn=None,
                    nsteps=3,
                    flatten_input=False,
                    flatten_output=False,
                    lossfn='xent',
                    fix_data=True,
                    cuda=True,
                    nseeds=1,
                    show_progress=True,
                    dataloader_map=None       # ★ 新增：按 width 存的 dataloader
                    ):
    import pandas as pd
    coord_data_list = []

    # 如果传了 dataloader_map，就忽略全局 dataloader / fix_data 这套逻辑
    use_per_width_loader = dataloader_map is not None

    if not use_per_width_loader:
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

            # ★ 这里按 width 选择自己的 dataloader
            if use_per_width_loader:
                local_loader = dataloader_map[width]
            else:
                local_loader = dataloader

            for batch_idx, batch in enumerate(local_loader, 1):
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
            model = model_fn(width=w, ntokens=ntokens, args=args)
            model = setprec(model, args.precision)
            return model
        return f

    widths = 128 * (np.arange(3, 13))**2
    models = {int(w): gen(int(w)) for w in widths}

    dataloader_map = {}
    for w in widths:
        dim = int(8 * (w / 128.0) ** 0.5)
        transformation = torchvision.transforms.Compose([
            torchvision.transforms.Resize(dim),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                             (0.5, 0.5, 0.5)),
        ])

        trainset_w = DatasetClass(
            root='data/', train=True, download=True, transform=transformation
        )
        trainloader_w = torch.utils.data.DataLoader(
            trainset_w, batch_size=batch_size, shuffle=True, num_workers=4
        )

        batch = next(iter(trainloader_w))
        train_data_w = [batch] * nsteps

        dataloader_map[w] = train_data_w

    for mode in ['bs', 'scale', 'both']:
        df = _get_coord_data(
            models,
            dataloader_map=dataloader_map,   # ★ 用 per-width 数据
            optimizer_fn=lambda net: optimizer_fn(net, args, len(trainset), mode=mode),
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
            save_to=None,
            suptitle=f'Coord Check ({mode})',
            face_color=None
        )

coord_check_split_terms(
    lr=args.lr,
    model_fn=my_mlp_fn,
    optimizer_fn=my_custom_optimizer_fn,
    batch_size=args.mini_bs,
    nsteps=3,
    nseeds=3,
    args=args
)
