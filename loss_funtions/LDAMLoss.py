import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        self.device = 'cpu'
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list, device=self.device)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        if target.data.dim() == 1:
            labels = target.data
            # print(labels)
            argmax_target = torch.argmax(labels.data.to(dtype=torch.int64), 0)
            # print(argmax_target)
            index.scatter_(0, argmax_target, 1)
            index_float = index.type(torch.FloatTensor)
            batch_m = torch.matmul(self.m_list[None, :], index_float)
            batch_m = batch_m.view((-1, 1))
            x_m = x - batch_m
            labels = labels[None, :]
        else:
            labels = target.data
            # print(labels)
            argmax_target = torch.argmax(labels.data.to(dtype=torch.int64), 1)
            index.scatter_(1, argmax_target.view(-1, 1), 1)
            index_float = index.type(torch.FloatTensor)
            batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
            batch_m = batch_m.view((-1, 1))
            x_m = x - batch_m


        output = torch.where(index, x_m, x)
        # print("output: ", (self.s * output).shape, "target: ", labels.shape, "weight: ", self.weight)
        # print("***********------------------************")
        return F.cross_entropy(self.s * output, labels, weight=self.weight)

    # def forward(self, x, target):
    #     index = torch.zeros_like(x, dtype=torch.uint8, device=self.device)
    #     # print(type(target), type(target.data))
    #     # print(target.data, torch.argmax(target.data, dim=0))
    #     # index.scatter_(1, target.data.to(self.device, torch.int64), 1)
    #     # labels = torch.argmax(target.data, dim=0)
    #     # index.scatter_(1, labels.view(-1, 1).to(torch.int64), 1)
    #     index_float = target.data.type(torch.cuda.FloatTensor)
    #     print(index_float.dim(), index_float)
    #     if index_float.dim() == 1:
    #         index_float = index_float[:, None].to(self.device)
    #         print(index_float.dim(), index_float)
    #         target = target[:, None].to(self.device)
    #     else:
    #         index_float = index_float.transpose(0, 1).to(self.device)
    #     # print(self.m_list[None, :], index_float.transpose(0, 1).to(self.device))
    #     batch_m = torch.matmul(self.m_list[None, :], index_float)
    #     batch_m = batch_m.view((-1, 1))
    #     x_m = x - batch_m
    #
    #     output = torch.where(index, x_m, x)
    #     return F.cross_entropy(self.s * output, target, weight=self.weight)