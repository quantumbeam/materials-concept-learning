import torch
import torch.nn.functional as F
import torch.nn as nn

from torch_geometric.nn import global_max_pool, global_mean_pool

def normalize_embedding(x, mode):
    mode = mode.lower()
    if mode == 'l1':
        x = F.normalize(x, p=1, dim=1, eps=1e-6)
    elif mode == 'l2':
        x = F.normalize(x, p=2, dim=1, eps=1e-6)
    elif mode == 'sqrtd':
        scale = float(x.shape[1])**(-0.5)
        x = x * scale
    elif mode == "no" or mode is None:
        pass
    else:
        raise NameError(f"embedding_normalize: {mode}")
        
    return x


def MLP(channels):
    return nn.Sequential(*[
        nn.Sequential(
            nn.Linear(channels[i - 1], channels[i]),
            nn.BatchNorm1d(channels[i]),
            nn.ReLU())
        for i in range(1, len(channels))
    ])

def normalize_scale(pos, batch, scale=None):
    # Center
    mean = global_mean_pool(pos, batch)
    pos = pos - mean[batch]

    # Scale to [-1, 1]
    if scale is None:
        abs_max = global_max_pool(pos.abs(), batch).max(dim=1)[0] + 1e-6
        scale = 1.0/abs_max
    pos = pos * scale[batch][:, None]
    return pos, scale
