import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from models import basicblock as B


def sum(x, device):
    pi = torch.tensor(math.pi)
    w = x.size()[-2]
    h = x.size()[-1]
    eh = 6.0 * (w - 2.0) * (h - 2.0)

    r = noise_esti(x, device)
    sr = torch.sum(torch.abs(r), (2, 3))[0]
    sumr = 2 * (torch.sqrt(pi / 2.0) * (1.0 / eh)) * (sr)
    return sumr


def noise_esti(x, device):
    a = [1, -2, 1, -2, -4, -2, 1, -2, 1]
    kernel = torch.tensor(a).reshape(1, 1, 3, 3).float().to(device)
    kernel = kernel.repeat(x.shape[1], 1, 1, 1)
    b = F.conv2d(input=x, weight=kernel, stride=3, padding=1)
    return b


class IRCNN(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=32):
        super(IRCNN, self).__init__()
        self.model = B.IRCNN(in_nc, out_nc, nc)
        self.device = torch.device('cuda')

    def forward(self, x):
        n = self.model(x)
        level = n
        return n, level
