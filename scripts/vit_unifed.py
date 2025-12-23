'''Train CIFAR10/CIFAR100 with PyTorch.'''
from .logger import ExecutionLogger
import numpy as np
import torch.nn.functional as F
from .logger import ExecutionLogger
from .get_params import get_shapes, _get_noise4target, _get_lr4target, _get_clip4target
from .model_builder import MyVit, MyPreVit



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
    # model_base = MyVit(args, is_base=True)
    # base_model = model_base.create_model() 
    # base_shapes = get_shapes(base_model)
    
    # model_target = MyVit(args, is_base=False)    
    # net = model_target.create_model()
    # net.apply(kaiming_init_weights)
    # model_shapes = get_shapes(net)

    model_base = MyPreVit(args, is_base=True)
    base_model = model_base.create_model() 
    model_target = MyPreVit(args, is_base=False)
    net = model_target.create_model()
    net.apply(kaiming_init_weights)
    model_shapes = get_shapes(net)
    
    noise = _get_noise4target(base_shapes, model_shapes, base_noise=args.noise)
    clip_dict = _get_clip4target(base_shapes, model_shapes, target_noise=noise)
    D_prime_vector = torch.stack(list(clip_dict.values()))
    print(clip_dict)

    net = ModuleValidator.fix(net)
    net = net.to(device)

    print('Number of total parameters: ', sum([p.numel() for p in net.parameters()]))
    print('Number of trainable parameters: ', sum([p.numel() for p in net.parameters() if p.requires_grad]))

    criterion = F.cross_entropy

    base_lr = 2 ** args.lr
    # target_lr_dict = _get_lr4target(model_shapes, noise/args.bs, base_lr)
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
      
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(param_groups, lr=base_lr)


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
import warnings; 
warnings.filterwarnings("ignore")


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
