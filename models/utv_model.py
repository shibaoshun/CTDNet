import torch
import torch.nn as nn
# -*- coding: utf-8 -*-

import torch.fft
import torch
import torch.nn as nn
from models import basicblock as B
import numpy as np
from models import nli_model

class ADMM(nn.Module):
    def __init__(self, inc, k, cha):
        super(ADMM, self).__init__()
        self.device = torch.device('cuda')
        self.hyp = B.HyPaNet(inc, k, cha)
        self.ouc = k
        self.mlp = nn.Sequential()
        # self.noiselevel = nli_model.IRCNN(1, 1, 32)
        self.rou = nn.Parameter(torch.FloatTensor([15]), requires_grad=True)
        self.lamda = nn.Parameter(torch.FloatTensor([0.1]), requires_grad=True)
        self.rout = nn.Parameter(torch.FloatTensor([0.1]), requires_grad=True)
    def fftn(self, t, row, col, dim):
        y = torch.fft.fftn(t, col, dim=dim)
        y = y.expand(col, row)
        return y

    def fftnt(self, t, row, col, dim):
        y = torch.fft.fftn(t, col, dim=dim)
        y = y.expand(row, col)
        return y

    def ForwardDiff(self, x):
        x_diff = x[:, :, 1:] - x[:, :, :-1]
        x_e = (x[:, :, 0] - x[:, :, -1]).unsqueeze(2)
        x_diff = torch.cat((x_diff, x_e), 2)

        y_diff = x[:, 1:, :] - x[:, :-1, :]
        y_e = (x[:, 0, :] - x[:, -1, :]).unsqueeze(1)
        y_diff = torch.cat((y_diff, y_e), 1)

        return x_diff, y_diff

    def Dive(self, x, y):
        x_diff = x[:, :, :-1] - x[:, :, 1:]
        x_e = (x[:, :, -1] - x[:, :, 0]).unsqueeze(2)
        x_diff = torch.cat((x_e, x_diff), 2)


        y_diff = y[:, :-1, :] - y[:, 1:, :]
        y_e = (y[:, -1, :] - y[:, 0, :]).unsqueeze(1)
        y_diff = torch.cat((y_e, y_diff), 1)

        return y_diff + x_diff

    def shrink(self, x, r, m):
        z = torch.sign(x) * torch.max(torch.abs(x) - r, m)
        return z

    def forward(self, yo):
        batch, row, col = yo.size()[0], yo.size()[1], yo.size()[2]
        y = yo
        v1 = torch.zeros(batch, row, col).to(self.device)
        v2 = torch.zeros(batch, row, col).to(self.device)
        m = torch.zeros(batch, row, col).to(self.device)
        y1 = torch.zeros(batch, row, col).to(self.device)
        y2 = torch.zeros(batch, row, col).to(self.device)

        x1 = ([1.0], [-1.0])
        x2 = ([1.0, -1.0])

        Dx = torch.tensor(x1).to(self.device)
        x3 = torch.tensor(x2).to(self.device)

        eigDtD = torch.pow(torch.abs(self.fftn(Dx, col, row, 0)), 2) + torch.pow(torch.abs(self.fftnt(x3, row, col, 0)),
                                                                                 2).to(self.device)
        x=y
        for k in range(0, self.ouc):
            rhs = 2*self.rout*y - self.rou * self.Dive(y1/self.rou - v1, y2/self.rou - v2)
            lhs = 2*self.rout + self.rou * (eigDtD)
            lhs = torch.unsqueeze(lhs, 0)
            lhs = lhs.repeat(yo.size()[0], 1, 1)

            x = torch.div(torch.fft.fftn(rhs), lhs)
            x = torch.real(torch.fft.ifftn(x))
            Dx1, Dx2 = self.ForwardDiff(x)
            u1 = Dx1 + y1/self.rou
            u2 = Dx2 + y2/self.rou
            v1 = self.shrink(u1, self.lamda/self.rou, m)
            v2 = self.shrink(u2,  self.lamda/self.rou, m)
            y1 = y1 - self.rou * (v1 - Dx1)
            y2 = y2 - self.rou * (v2 - Dx2)

        return x
