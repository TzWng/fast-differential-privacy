# @title '''Train CIFAR10/CIFAR100 with PyTorch.'''

import numpy as np
import torch.nn.functional as F

from fastDP import PrivacyEngine 
import math, torch, os, torchvision, timm
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
from torchvision import datasets, transforms 
from opacus.validators import ModuleValidator 
from opacus.accountants.utils import get_noise_multiplier 
from torch import nn 
from tqdm import tqdm 

from .logger import ExecutionLogger
from .get_params import get_shapes, _get_noise4target, _get_lr4target, _get_clip4target, _get_lr4target_adam


import warnings; 
warnings.filterwarnings("ignore")


class MLP(nn.Module):
    def __init__(self, width=128, input_dim=3072, num_classes=10, nonlin=F.relu, output_mult=1.0, input_mult=1.0):
        super(MLP, self).__init__()
        self.nonlin = nonlin
        self.input_mult = input_mult
        self.output_mult = output_mult
        self.fc_1 = nn.Linear(input_dim, width, bias=False)
        self.fc_2 = nn.Linear(width, width, bias=False)
        self.fc_3 = nn.Linear(width, width, bias=False)
        self.fc_4 = nn.Linear(width, width, bias=False)
        self.fc_5 = nn.Linear(width, num_classes, bias=False) 
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.fc_1.weight, a=1, mode='fan_in')
        self.fc_1.weight.data /= self.input_mult**0.5
        nn.init.kaiming_normal_(self.fc_2.weight, a=1, mode='fan_in')
        nn.init.kaiming_normal_(self.fc_3.weight, a=1, mode='fan_in')
        nn.init.kaiming_normal_(self.fc_4.weight, a=1, mode='fan_in')
        # nn.init.kaiming_normal_(self.fc_5.weight, a=1, mode='fan_in')
        # self.fc_5.weight.data /= self.output_mult
        nn.init.zeros_(self.fc_5.weight)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        out = self.nonlin(self.fc_1(x) * self.input_mult**0.5)
        out = self.nonlin(self.fc_2(out))
        out = self.nonlin(self.fc_3(out))
        out = self.nonlin(self.fc_4(out))
        return self.fc_5(out) * self.output_mult


def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T

    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(0) > G.size(1):
        X = X.T
    return X


import torch
import torch.optim as optim
import math

class MuonNEW(optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, 
                 ns_steps=6, head_param_ids=None,
                 adam_betas=(0.95, 0.95), adam_eps=1e-8):
        """
        Args:
            params: Model parameters
            lr: Learning rate
            momentum: Momentum for Muon part
            nesterov: Nesterov flag for Muon part
            ns_steps: Newton-Schulz iteration steps
            head_param_ids: A set(id(p)) marking which parameters belong to the Head
            adam_betas: Betas for Adam (Head only)
            adam_eps: Epsilon for Adam (Head only)
        """
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, 
                        adam_betas=adam_betas, adam_eps=adam_eps)
        
        super().__init__(params, defaults)
        
        self.head_param_ids = set() if head_param_ids is None else head_param_ids
        self._head_param_set = set()

        # Pre-processing
        for group in self.param_groups:
            for p in group["params"]:
                if id(p) in self.head_param_ids:
                    self._head_param_set.add(p)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]
            nesterov = group["nesterov"]
            
            # Adam specific params
            beta1, beta2 = group["adam_betas"]
            eps = group["adam_eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                g = p.grad

                # ==================================================
                # Branch 1: Handle Head parameters (Adam + Custom Scaling)
                # ==================================================
                if p in self._head_param_set:
                    state = self.state[p]

                    # 状态初始化
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    state['step'] += 1

                    # 1. 计算 Adam 的一阶和二阶矩 (No Weight Decay)
                    exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                    # 2. Bias Correction
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    
                    # 3. 计算自定义 Scaling Factor
                    scaling_factor = 1.0
                    if g.ndim == 2:
                        spec = g.size(0) ** 0.5 + g.size(1) ** 0.5
                        scaling_factor = (g.size(0) / g.size(1)) ** 0.5 / spec

                    # 4. 组合最终的 step size
                    # step_size = lr * scale * (sqrt(1-beta2^t) / (1-beta1^t))
                    step_size = (lr * scaling_factor) / bias_correction1

                    # Update parameters
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # ==================================================
                # Branch 2: Handle Muon parameters (Original Logic Unchanged)
                # ==================================================
                else:
                    if g.ndim > 2:
                        g = g.view(g.size(0), -1)
                    
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    else:
                        g = buf
                    
                    if g.ndim >= 2:
                        g = zeropower_via_newtonschulz5(g, steps=ns_steps)
                        g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    else:
                        g /= (g.norm() + 1e-8)

                    p.data.add_(g, alpha=-lr)

        return loss


def main(args):
    if args.clipping_mode not in ['nonDP', 'BK-ghost', 'BK-MixGhostClip', 'BK-MixOpt', 'nonDP-BiTFiT', 'BiTFiT']:
        print("Mode must be one of 'nonDP','BK-ghost', 'BK-MixGhostClip', 'BK-MixOpt','nonDP-BiTFiT','BiTFiT'")
        return None

    device = torch.device("cuda:0")

    # Data
    print('==> Preparing data..')

    transformation = torchvision.transforms.Compose([
        torchvision.transforms.Resize(args.dimension),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if args.cifar_data == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root='data/', train=True, download=True, transform=transformation)
        testset = torchvision.datasets.CIFAR10(root='data/', train=False, download=True, transform=transformation)
    elif args.cifar_data == 'CIFAR100':
        trainset = torchvision.datasets.CIFAR100(root='data/', train=True, download=True, transform=transformation)
        testset = torchvision.datasets.CIFAR100(root='data/', train=False, download=True, transform=transformation)
    else:
        return "Must specify datasets as CIFAR10 or CIFAR100"
    

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.mini_bs, shuffle=True, num_workers=4)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=4)

    n_acc_steps = args.bs // args.mini_bs  # gradient accumulation steps

    # Model
    print('==> Building model MLP; BatchNorm is replaced by GroupNorm. Mode: ', args.clipping_mode)
    input_dim = 3 * args.dimension * args.dimension
    
    # net = FlexibleMLP(width=args.width, input_dim=input_dim, num_layers=args.layer, nonlin=torch.relu, output_mult=32, input_mult=1/256).to(device)
    base_model = MLP(width=128, input_dim=input_dim, nonlin=torch.relu, output_mult=32, input_mult=1/256)
    net = MLP(width=args.width, input_dim=input_dim, nonlin=torch.relu, output_mult=32, input_mult=1/256).to(device)
    base_shapes = get_shapes(base_model)
    model_shapes = get_shapes(net)
    noise = _get_noise4target(base_shapes, model_shapes, base_noise=args.noise)
    clip_dict = _get_clip4target(base_shapes, model_shapes, target_noise=noise)
    D_prime_vector = torch.stack(list(clip_dict.values()))
    
    print('Number of total parameters: ', sum([p.numel() for p in net.parameters()]))
    print('Number of trainable parameters: ', sum([p.numel() for p in net.parameters() if p.requires_grad]))
    
    criterion = F.cross_entropy

    base_lr = 2 ** args.lr
      
    if args.optimizer == 'SGD':
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
        # optimizer = optim.SGD(net.parameters(), lr=base_lr)
    elif args.optimizer == 'Adam':
        target_lr_dict = _get_lr4target_adam(base_shapes, model_shapes, args.noise, noise, base_lr)
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
        optimizer = optim.Adam(param_groups, lr=base_lr)
    elif args.optimizer == 'muon':
        head_ids = {id(p) for p in net.fc_1.parameters()} | {id(p) for p in net.fc_5.parameters()}
        optimizer = MuonNEW(net.parameters(), lr=base_lr, momentum=0.95, nesterov=True, ns_steps=6, head_param_ids=head_ids)
        

    if 'BiTFiT' in args.clipping_mode:  # not needed for DP-BiTFiT but use here for safety
        for name, param in net.named_parameters():
            if '.bias' not in name:
                param.requires_grad_(False)

    # Privacy engine
    if 'nonDP' not in args.clipping_mode:
        sigma = get_noise_multiplier(
            target_epsilon=args.epsilon,
            target_delta=1e-5,
            sample_rate=args.bs / len(trainset),
            epochs=args.epochs,
        )
        print("epsilon delta noise is", sigma)
        
        if 'BK' in args.clipping_mode:
            clipping_mode = args.clipping_mode[3:]
        else:
            clipping_mode = 'ghost'

        if args.clipping_style in [['all-layer'], ['layer-wise'], ['param-wise']]:
            args.clipping_style = args.clipping_style[0]
        privacy_engine = PrivacyEngine(
            net,
            batch_size=args.bs,
            sample_size=len(trainset),
            noise_multiplier=noise,
            epochs=args.epochs,
            clipping_mode=clipping_mode,
            clipping_coe=D_prime_vector,
            clipping_style=args.clipping_style,
            origin_params=args.origin_params,  # ['patch_embed.proj.bias'],
        )
        privacy_engine.attach(optimizer)
        print("Noise multiplier (σ):", privacy_engine.noise_multiplier)

    def train(epoch):

        net.train()
        train_loss = 0
        correct = 0
        total = 0
        eps = 1e-6

        for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.view(inputs.size(0), -1)

            # forward + loss
            outputs = net(inputs)
            loss = criterion(outputs, targets) / n_acc_steps
            loss.backward()
            
            if ((batch_idx + 1) % n_acc_steps == 0) or ((batch_idx + 1) == len(trainloader)):
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * n_acc_steps
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


        print('Epoch: ', epoch, len(trainloader), 'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        return train_loss / (batch_idx + 1)

    def test(epoch):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            print('Epoch: ', epoch, len(testloader), 'Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            
            return test_loss / (batch_idx + 1)

    for epoch in range(args.epochs):
        train_loss = train(epoch)
        # test_loss = test(epoch)
        if math.isnan(train_loss):
            break

    logger = ExecutionLogger(args.log_path)
    # logger.log(log2lr=args.lr, train_loss=train_loss, depth=args.layer, batch=args.bs, sigma=args.noise)
    logger.log(log2lr=args.lr, train_loss=train_loss, width=args.width, batch=args.bs, sigma=noise)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
    parser.add_argument('--width', default=256, type=int)
    parser.add_argument('--layer', default=3, type=int)   
    parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--bs', default=512, type=int)
    parser.add_argument('--mini_bs', type=int, default=512)
    parser.add_argument('--epsilon', default=2, type=float)
    parser.add_argument('--noise', default=1, type=float)
    parser.add_argument('--seed', default=4, type=int)
    parser.add_argument('--clipping_mode', default='BK-ghost', type=str)
    parser.add_argument('--clipping_style', default='layer-wise', nargs='+', type=str)
    parser.add_argument('--scale', default=1, type=int)
    parser.add_argument('--cifar_data', type=str, default='CIFAR10')
    parser.add_argument('--dimension', type=int, default=32)
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--origin_params', nargs='+', default=None)
    parser.add_argument(
        '--log_path',
        type=str,
        default='/content/drive/MyDrive/DP_muP/logs/MLP_Adam_DP_noise.txt',
    )

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    main(args)
