import torch
import torch.nn.functional as F
import torch.nn as nn

def regression_loss(pred, data, targets):
    assert(len(targets)==pred.shape[1])

    loss = 0
    for i, t in enumerate(targets):
        labels = data[t]
        loss += F.l1_loss(pred[:, i], labels)
    return loss
