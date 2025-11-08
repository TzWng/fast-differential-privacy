# @title '''Train CIFAR10/CIFAR100 with PyTorch.'''

import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, width=128, input_dim=3072, num_classes=10, nonlin=F.relu, output_mult=1.0, input_mult=1.0):
        super(MLP, self).__init__()
        self.nonlin = nonlin
        self.input_mult = input_mult
        self.output_mult = output_mult
        self.fc_1 = nn.Linear(input_dim, width, bias=False)
        self.fc_2 = nn.Linear(width, width, bias=False)
        self.fc_3 = nn.Linear(width, num_classes, bias=False)
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.kaiming_normal_(self.fc_1.weight, a=1, mode='fan_in')
        self.fc_1.weight.data /= self.input_mult**0.5
        nn.init.kaiming_normal_(self.fc_2.weight, a=1, mode='fan_in')
        nn.init.zeros_(self.fc_3.weight)


    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        out = self.nonlin(self.fc_1(x) * self.input_mult**0.5)
        out = self.nonlin(self.fc_2(out))
        return self.fc_3(out) * self.output_mult


    
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
    net = MLP(width=args.width, input_dim=input_dim, nonlin=torch.relu, output_mult=32, input_mult=1/256).to(device)

        
    print('Number of total parameters: ', sum([p.numel() for p in net.parameters()]))
    print('Number of trainable parameters: ', sum([p.numel() for p in net.parameters() if p.requires_grad]))
    
    criterion = F.cross_entropy

    base_lr = 2 ** args.lr
    param_groups = [
        {"params": [p], "lr": base_lr, "name": n}
        for n, p in net.named_parameters()
    ]

    optimizer = optim.SGD(param_groups, lr=base_lr)

    
    sigma = get_noise_multiplier(
        target_epsilon=args.epsilon,
        target_delta=1e-5,
        sample_rate=args.bs / len(trainset),
        epochs=args.epochs,
        )
    
    clip_value = 1.0    


    def train(epoch):

        net.train()
        optimizer.zero_grad()
        train_loss = 0
        correct = 0
        total = 0
        eps = 1e-6

        # optimizer.zero_grad()

        for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.view(inputs.size(0), -1)

            # forward + loss
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            if ((batch_idx + 1) % n_acc_steps == 0) or ((batch_idx + 1) == len(trainloader)):
                
                # first clipping + noise
                # optimizer.step()

                for group in optimizer.param_groups:
                    param = group["params"][0]
                    # n_out, n_in = param.shape[0], param.shape[1]
                    # Cl = clip_value * math.sqrt(n_out / n_in)

                    
                    # current_norm = torch.norm(param.grad.detach(), p=2).item()
                    # clip_coef = min(1.0, Cl / (current_norm + eps))
                    
                    # param.grad.mul_(clip_coef)

                    # grad = param.grad.detach().clone()
                    # clip_norm = torch.linalg.norm(grad, ord=2).item()
                    # n_out, n_in = grad.shape[0], grad.shape[1]
                    # noise_spec_approx = (math.sqrt(n_out) + math.sqrt(n_in)) * (sigma * clip_value)
                    # denom = math.sqrt(clip_norm**2 + noise_spec_approx**2)


                    # noise = torch.normal(mean=0, std=sigma*Cl,
                    #                      size=param.grad.shape, device=param.grad.device,
                    #                     )
                    # param.grad += noise
                    grad = param.grad

                    lr_scale = 1.0
                    if grad is not None and grad.ndim in (1, 2):
                        spec = torch.linalg.norm(grad, ord=2).clamp(min=eps)
                        if grad.ndim == 2:
                            lr_scale = (param.shape[0] / param.shape[1]) ** 0.5 / spec
                        else:
                            lr_scale = (param.shape[0]) ** 0.5 / spec
                    group["lr"] = base_lr * lr_scale


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

    os.makedirs("/content/drive/MyDrive/DP_muP/logs", exist_ok=True)
    with open(args.log_path, "a") as f:
        f.write(f"log2lr = {args.lr:.4f}, train_loss = {train_loss:.4f}, width = {args.bs}\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
    parser.add_argument('--width', default=256, type=int)
    parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
    parser.add_argument('--epochs', default=20, type=int,
                        help='numter of epochs')
    parser.add_argument('--bs', default=512, type=int, help='batch size')
    parser.add_argument('--mini_bs', type=int, default=512)
    parser.add_argument('--epsilon', default=2, type=float, help='target epsilon')
    parser.add_argument('--clipping_mode', default='BK-MixOpt', type=str)
    parser.add_argument('--clipping_style', default='layer-wise', nargs='+', type=str)
    parser.add_argument('--scale', default=1, type=int)
    parser.add_argument('--cifar_data', type=str, default='CIFAR10')
    parser.add_argument('--dimension', type=int, default=32)
    parser.add_argument('--origin_params', nargs='+', default=None)
    parser.add_argument('--log_path', type=str, default='/content/drive/MyDrive/DP_muP/logs/MLP_Adam_DP_noise.txt',
                        help='Path to save training log')

    args = parser.parse_args()

    from fastDP import PrivacyEngine

    import math
    import torch
    import os
    import torchvision

    torch.manual_seed(2)
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

    main(args)
