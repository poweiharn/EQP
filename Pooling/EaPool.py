import torch
import torch.nn as nn
import torch.nn.functional as F
from Pooling.get_pooling import get_pooling


class EaPool(nn.Module):
    def __init__(self, kernel_size, stride, gene, cuda, padding=0):
        super(EaPool, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.gene = gene
        self.cuda = cuda

    def window_mapping(self, x):
        _, c, h, w = x.size()
        y = torch.zeros([_, c, h // 2, w // 2])
        if self.cuda:
            y = y.cuda()
        k = 0
        for i in range(h // 2):
            start_h = 2 * i
            end_h = start_h + 2
            for j in range(w // 2):
                start_w = 2 * j
                end_w = start_w + 2
                pool = get_pooling({"pooling": self.gene[k]})
                p = pool(self.kernel_size, self.stride)
                y[:, :, start_h // 2: end_h // 2, start_w // 2: end_w // 2] = p(x[:, :, start_h:end_h, start_w:end_w])
                k = k + 1
        return y

    def forward(self, x):
        x = self.window_mapping(x)
        return x