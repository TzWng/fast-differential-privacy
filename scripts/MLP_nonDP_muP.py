# @title '''Train CIFAR10/CIFAR100 with PyTorch.'''

import torch.nn as nn
import torch.nn.functional as F
from .logger import ExecutionLogger
import numpy as np
import torch
from torch.utils.data import DataLoader

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

class muMLP(nn.Module):
    def __init__(self, width=128, input_dim=3072, num_classes=10, nonlin=F.relu, output_mult=1.0, input_mult=1.0):
        super(muMLP, self).__init__()
        self.nonlin = nonlin
        self.input_mult = input_mult
        self.output_mult = output_mult
        self.fc_1 = nn.Linear(input_dim, width, bias=False)
        self.fc_2 = nn.Linear(width, width, bias=False)
        self.fc_3 = nn.Linear(width, width, bias=False)
        self.fc_4 = nn.Linear(width, width, bias=False)
        self.fc_5 = MuReadout(width, num_classes, bias=False, output_mult=self.output_mult)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.fc_1.weight, a=1, mode='fan_in')
        self.fc_1.weight.data /= self.input_mult ** 0.5
        nn.init.kaiming_normal_(self.fc_2.weight, a=1, mode='fan_in')
        nn.init.kaiming_normal_(self.fc_3.weight, a=1, mode='fan_in')
        nn.init.kaiming_normal_(self.fc_4.weight, a=1, mode='fan_in')
        nn.init.zeros_(self.fc_5.weight)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        out = self.nonlin(self.fc_1(x) * self.input_mult**0.5)
        out = self.nonlin(self.fc_2(out))
        out = self.nonlin(self.fc_3(out))
        out = self.nonlin(self.fc_4(out))
        return self.fc_5(out)


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
        
class MuonNEW(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=6):

        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)

        super().__init__(params, defaults)

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']

            # generate weight updates in distributed fashion
            for i, p in enumerate(group['params']):
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if group['nesterov']:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                    
                if g.ndim >= 2:
                    g = zeropower_via_newtonschulz5(g, steps=group['ns_steps'])
                    g *= max(1, g.size(0)/g.size(1))**0.5
                else:
                    g /= g.norm()
                p.data.add_(g, alpha=-lr)

        return loss


def main(args):
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

    # # Model
    # input_dim = 3 * args.dimension * args.dimension
    # net = MLP(width=args.width, input_dim=input_dim, nonlin=torch.relu, output_mult=32, input_mult=1/256).to(device)

    
    input_dim = 3 * args.dimension * args.dimension
    base_model = MLP(width=128, input_dim=input_dim, nonlin=torch.relu, output_mult=32, input_mult=1/256)
    delta_model = MLP(width=256, input_dim=input_dim, nonlin=torch.relu, output_mult=32, input_mult=1/256)
    net = muMLP(width=args.width, input_dim=input_dim, nonlin=torch.relu, output_mult=32, input_mult=1/256).to(device)
    net = net.to(device)
    set_base_shapes(net, base_model, delta=delta_model)

    

        
    print('Number of total parameters: ', sum([p.numel() for p in net.parameters()]))
    print('Number of trainable parameters: ', sum([p.numel() for p in net.parameters() if p.requires_grad]))
    
    criterion = F.cross_entropy

    base_lr = 2 ** args.lr

    param_groups = [
        {"params": [p], "lr": base_lr, "name": n}
        for n, p in net.named_parameters()
    ]

    if args.optimizer == 'SGD':
        optimizer = MuSGD(net.parameters(), lr=base_lr)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(param_groups, lr=base_lr)
    elif args.optimizer == 'muon':
        optimizer = MuonNEW(net.parameters(), lr=base_lr, momentum=0.95, nesterov=True, ns_steps=5)

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
                
                # for group in optimizer.param_groups:
                #     name = group.get("name", "")
                #     param = group["params"][0]
                #     grad = param.grad
                #     lr_scale = 1.0                  
                   
                #     if grad is not None and grad.ndim in (1, 2):                               
                #         if grad.ndim == 2:
                #             if args.optimizer == 'SGD':
                #                 lr_scale = param.shape[0] / param.shape[1]
                #             elif args.optimizer == 'Adam':
                #                 a = (param.shape[0] ** 0.5 + param.shape[1] ** 0.5)
                #                 lr_scale = 1 / param.shape[1]
                #         elif grad.ndim == 1:
                #             lr_scale = (param.shape[0]) ** 0.5 / spec
                            
                #     group["lr"] = base_lr * lr_scale

                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * n_acc_steps
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


        print('Epoch: ', epoch, len(trainloader), 'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        return train_loss / (batch_idx + 1)

    for epoch in range(args.epochs):
        train_loss = train(epoch)
        if math.isnan(train_loss):
            break

    logger = ExecutionLogger(args.log_path)
    # logger.log(log2lr=args.lr, train_loss=train_loss, depth=args.layer, batch=args.bs, sigma=args.noise)
    logger.log(log2lr=args.lr, train_loss=train_loss, width=args.width, batch=args.bs, sigma=args.noise)


import mup
from mup import MuSGD, get_shapes, set_base_shapes, make_base_shapes, MuReadout
import math, torch, os, torchvision 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
from torchvision import datasets, transforms 
from torch import nn 
from tqdm import tqdm 
import warnings; 
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
    parser.add_argument('--width', default=256, type=int)
    parser.add_argument('--layer', default=3, type=int)   
    parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--bs', default=512, type=int)
    parser.add_argument('--mini_bs', type=int, default=512)
    parser.add_argument('--noise', default=0, type=float)
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
    torch.manual_seed(4)
    main(args)
