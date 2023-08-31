import torch
from torch import nn
from kornia.losses import FocalLoss


ALPHA = 0.8
GAMMA = 2


class FocalLossKornia(nn.Module):
    def __init__(self, alpha: float, gamma: float = 2.0, reduction: str = 'none', eps=None, **kwargs):
        self.params = kwargs
        super(FocalLossKornia, self).__init__()
        self.loss = FocalLoss(alpha, gamma, reduction, eps)

    def forward(self, y_pred, label):
        if y_pred.dim() == 1:
            label = torch.argmax(label, 0, keepdim=True).to(torch.int64)
            loss = self.loss(y_pred[None, :, None], label[None, :])
        else:
            label = torch.argmax(label, 1).to(torch.int64)
            loss = self.loss(y_pred[:, :, None], label[:, None])
        return torch.squeeze(loss).sum()
