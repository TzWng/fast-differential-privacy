# coord_check_mlp.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# MLP architecture
# ============================================================
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
        self.fc_1.weight.data /= self.input_mult ** 0.5
        nn.init.kaiming_normal_(self.fc_2.weight, a=1, mode='fan_in')
        nn.init.zeros_(self.fc_3.weight)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        out = self.nonlin(self.fc_1(x) * self.input_mult ** 0.5)
        out = self.nonlin(self.fc_2(out))
        return self.fc_3(out) * self.output_mult


# ============================================================
# Coordinate recording utilities
# ============================================================
def _record_coords(df, width, name, batch_idx):
    def hook(module, inputs, outputs):
        record = {
            'width': width,
            'module': name,
            'batch_idx': batch_idx,
            'activation_std': outputs.std().item() if isinstance(outputs, torch.Tensor) else None,
            'activation_mean': outputs.mean().item() if isinstance(outputs, torch.Tensor) else None,
            'activation_l1': outputs.abs().mean().item() if isinstance(outputs, torch.Tensor) else None,
        }
        df.append(record)
    return hook


def _get_coord_data(models, dataloader, optimizer_fn, nsteps=3, lossfn='xent', cuda=True, nseeds=1):
    coord_data_list = []
    for i in range(nseeds):
        torch.manual_seed(i)
        for width, model_ctor in models.items():
            model = model_ctor()
            if cuda:
                model = model.cuda()
            optimizer = optimizer_fn(model)
            model.train()
            for batch_idx, (data, target) in enumerate(dataloader, 1):
                remove_hooks = []
                for name, module in model.named_modules():
                    remove_hooks.append(module.register_forward_hook(_record_coords(coord_data_list, width, name, batch_idx)))
                if cuda:
                    data, target = data.cuda(), target.cuda()
                out = model(data)
                loss = F.cross_entropy(out, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                for h in remove_hooks:
                    h.remove()
                if batch_idx == nsteps:
                    break
    return pd.DataFrame(coord_data_list)


# ============================================================
# Main: Coordinate Check
# ============================================================
def coord_check_split_terms(model_fn, optimizer_fn, batch_size, nsteps, nseeds, trainset):
    widths = 2 ** np.arange(7, 11)  # widths: 128–1024
    models = {w: (lambda w=w: model_fn(w)) for w in widths}

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    batch = next(iter(trainloader))
    fixed_data = [batch] * nsteps  # same batch repeated

    df = _get_coord_data(models, dataloader=fixed_data, optimizer_fn=optimizer_fn, nsteps=nsteps, nseeds=nseeds)

    # plotting
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    for w in sorted(df['width'].unique()):
        subset = df[df['width'] == w]
        plt.plot(subset['batch_idx'], subset['activation_l1'], label=f'width={w}')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Batch')
    plt.ylabel('L1 activation')
    plt.legend()
    plt.title('Coordinate Check (MLP, CIFAR10)')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    import torch.optim as optim
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform)

    def model_fn(width):
        return MLP(width=width, input_dim=3 * 32 * 32, num_classes=10, nonlin=torch.relu, output_mult=32, input_mult=1/256)

    def optimizer_fn(model):
        return optim.SGD(model.parameters(), lr=1e-3)

    coord_check_split_terms(model_fn, optimizer_fn, batch_size=64, nsteps=3, nseeds=3, trainset=trainset)
