'''Train CIFAR10/CIFAR100 with PyTorch.'''

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
from .get_params import get_shapes, _get_noise4target, _get_lr4target, _get_clip4target
from .model_builder import MyVit, MyPreVit



import warnings; 
warnings.filterwarnings("ignore")



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

class MuonNEW(optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, 
                 ns_steps=6, noise=1.0, bs=256, head_param_ids=None):
        """
        Args:
            params: Model parameters
            lr: Learning rate
            momentum: Momentum for Muon part (Not used for Head)
            nesterov: Nesterov flag for Muon part
            ns_steps: Newton-Schulz iteration steps
            noise: Noise coefficient for Head scaling calculation
            bs: Batch Size (Used for Head scaling calculation)
            head_param_ids: A set(id(p)) marking which parameters belong to the Head
        """
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, 
                        ns_steps=ns_steps, noise=noise, bs=bs)
        
        super().__init__(params, defaults)
        
        self.head_param_ids = set() if head_param_ids is None else head_param_ids
        self._head_param_set = set()

        # Pre-processing: Identify specific Head parameter objects and store them 
        # in _head_param_set for fast lookup during the step
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
            # Get common hyperparameters
            lr = group["lr"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]
            nesterov = group["nesterov"]

            noise = group["noise"]
            bs = group["bs"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                g = p.grad

                # ==================================================
                # Branch 1: Handle Head parameters (Plain SGD, No Momentum)
                # ==================================================
                if p in self._head_param_set:
                    if g.ndim == 2:
                        spec = (g.size(0) ** 0.5 + g.size(1) ** 0.5) * noise / bs
                        lr_scale = (g.size(0) / g.size(1)) ** 0.5 / spec
                        g = g * lr_scale
                    
                    p.data.add_(g, alpha=-lr)

                # ==================================================
                # Branch 2: Handle Muon parameters (Newton-Schulz + Momentum)
                # ==================================================
                else:
                    # Flatten to 2D if dims > 2 (e.g., Conv2d)
                    if g.ndim > 2:
                        g = g.view(g.size(0), -1)
                    
                    # --- Momentum Handling ---
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    else:
                        g = buf
                    
                    # --- Newton-Schulz Orthogonalization ---
                    if g.ndim >= 2:
                        # Orthogonalize
                        g = zeropower_via_newtonschulz5(g, steps=ns_steps)
                        # Restore Scaling
                        g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    else:
                        # Simple normalization for 1D parameters (like bias)
                        g /= (g.norm() + 1e-8)

                    # Update weights
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

    if args.dataset == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root='data/', train=True, download=True, transform=transformation)
        testset = torchvision.datasets.CIFAR10(root='data/', train=False, download=True, transform=transformation)
    elif args.dataset == 'CIFAR100':
        trainset = torchvision.datasets.CIFAR100(root='data/', train=True, download=True, transform=transformation)
        testset = torchvision.datasets.CIFAR100(root='data/', train=False, download=True, transform=transformation)
    else:
        return "Must specify datasets as CIFAR10 or CIFAR100"

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.mini_bs, shuffle=True, num_workers=4)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=4)

    n_acc_steps = args.bs // args.mini_bs  # gradient accumulation steps


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


    # Model
    print('==> Building model..', args.model, '; BatchNorm is replaced by GroupNorm. Mode: ', args.clipping_mode)
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
    # target_lr_dict = _get_lr4target(model_shapes, noise/args.bs, base_lr)
    target_lr_dict = _get_lr4target(base_shapes, model_shapes, args.noise, noise, base_lr)
    

      
    if args.optimizer == 'SGD':
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
    
    
    elif args.optimizer == 'muon':
        head_ids = {id(p) for p in net.patch_embed.proj.parameters()} | {id(p) for p in net.head.parameters()}
        optimizer = MuonNEW(net.parameters(), lr=base_lr, momentum=0.95, nesterov=True, ns_steps=6,
                            noise=noise, head_param_ids=head_ids)


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

            # forward + loss
            outputs = net(inputs)
            loss = criterion(outputs, targets) / n_acc_steps
            loss.backward()
            
            if ((batch_idx + 1) % n_acc_steps == 0) or ((batch_idx + 1) == len(trainloader)):                
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item()
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
    logger.log(log2lr=args.lr, train_loss=train_loss, width=192*args.scale, batch=args.bs, sigma=noise)




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="vit_tiny_patch16_224")
    parser.add_argument("--lr", type=float, default=0)  # 和你画图保持一致
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--bs", type=int, default=512)
    parser.add_argument("--mini_bs", type=int, default=512)
    parser.add_argument("--epsilon", type=float, default=2.0)
    parser.add_argument('--noise', default=1, type=float)
    parser.add_argument("--clipping_mode", type=str, default="BK-ghost")
    parser.add_argument("--clipping_style", nargs="+", type=str, default="layer-wise")
    parser.add_argument('--scale', default=1, type=float)
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--dimension", type=int, default=224)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--origin_params", nargs="+", default=None)
    parser.add_argument(
        "--log_path",
        type=str,
        default="/content/drive/MyDrive/DP_muP/logs/MLP_Adam_DP_noise.txt",
    )


    args = parser.parse_args()
    torch.manual_seed(2)
    main(args)
