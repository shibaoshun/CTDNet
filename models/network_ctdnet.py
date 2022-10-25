import models.basicblock as B
import numpy as np
import torch.nn.functional as F
from .layers import *
from models import utv_model

"""
# --------------------------------------------
# basic functions
# --------------------------------------------
"""
def splits(a, sf):
    '''split a into sfxsf distinct blocks

    Args:
        a: NxCxWxHx2
        sf: split factor

    Returns:
        b: NxCx(W/sf)x(H/sf)x2x(sf^2)
    '''
    b = torch.stack(torch.chunk(a, sf, dim=2), dim=5)
    b = torch.cat(torch.chunk(b, sf, dim=3), dim=5)
    return b


def c2c(x):
    return torch.from_numpy(np.stack([np.float32(x.real), np.float32(x.imag)], axis=-1))


def r2c(x):
    # convert real to complex
    return torch.stack([x, torch.zeros_like(x)], -1)


def cdiv(x, y):
    # complex division
    a, b = x[..., 0], x[..., 1]
    c, d = y[..., 0], y[..., 1]
    cd2 = c**2 + d**2
    return torch.stack([(a*c+b*d)/cd2, (b*c-a*d)/cd2], -1)


def crdiv(x, y):
    # complex/real division
    a, b = x[..., 0], x[..., 1]
    return torch.stack([a/y, b/y], -1)


def csum(x, y):
    # complex + real
    return torch.stack([x[..., 0] + y, x[..., 1]], -1)


def cabs(x):
    # modulus of a complex number
    return torch.pow(x[..., 0]**2+x[..., 1]**2, 0.5)


def cabs2(x):
    return x[..., 0]**2+x[..., 1]**2


def cmul(t1, t2):
    '''complex multiplication

    Args:
        t1: NxCxHxWx2, complex tensor
        t2: NxCxHxWx2

    Returns:
        output: NxCxHxWx2
    '''
    real1, imag1 = t1[..., 0], t1[..., 1]
    real2, imag2 = t2[..., 0], t2[..., 1]
    return torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim=-1)


def cconj(t, inplace=False):
    '''complex's conjugation

    Args:
        t: NxCxHxWx2

    Returns:
        output: NxCxHxWx2
    '''
    c = t.clone() if not inplace else t
    c[..., 1] *= -1
    return c


def rfft(t):
    # Real-to-complex Discrete Fourier Transform
    return torch.rfft(t, 2, onesided=False)


def irfft(t):
    # Complex-to-real Inverse Discrete Fourier Transform
    return torch.irfft(t, 2, onesided=False)


def fft(t):
    # Complex-to-complex Discrete Fourier Transform
    return torch.fft(t, 2)


def ifft(t):
    # Complex-to-complex Inverse Discrete Fourier Transform
    return torch.ifft(t, 2)


def p2o(psf, shape):
    '''
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.

    Args:
        psf: NxCxhxw
        shape: [H, W]

    Returns:
        otf: NxCxHxWx2
    '''
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[...,:psf.shape[2],:psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)
    otf = torch.rfft(otf, 2, onesided=False)
    n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    otf[..., 1][torch.abs(otf[..., 1]) < n_ops*2.22e-16] = torch.tensor(0).type_as(psf)
    return otf


def upsample(x, sf=3):
    '''s-fold upsampler

    Upsampling the spatial size by filling the new entries with zeros

    x: tensor image, NxCxWxH
    '''
    st = 0
    z = torch.zeros((x.shape[0], x.shape[1], x.shape[2]*sf, x.shape[3]*sf)).type_as(x)
    z[..., st::sf, st::sf].copy_(x)
    return z


def downsample(x, sf=3):
    '''s-fold downsampler

    Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others

    x: tensor image, NxCxWxH
    '''
    st = 0
    return x[..., st::sf, st::sf]


def downsample_np(x, sf=3):
    st = 0
    return x[st::sf, st::sf, ...]


"""
# --------------------------------------------
# (1) Prior module: CTD Module
# --------------------------------------------
"""
class EBlock(nn.Module):
    def __init__(self, in_channel, out_channel, num_res=8, norm=False, first=False):
        super(EBlock, self).__init__()
        if first:
            layers = [Basic(in_channel, out_channel, kernel_size=3, norm=norm, relu=True, stride=1)]  # 步幅1卷积
        else:
            layers = [Basic(in_channel, out_channel, kernel_size=3, norm=norm, relu=True, stride=2)]  # 步幅2卷积

        layers += [ResBlock(out_channel, out_channel, norm) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8, norm=False, last=False, feature_ensemble=False):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel, norm) for _ in range(num_res)]

        if last:
            if feature_ensemble == False:
                layers.append(Basic(channel, 1, kernel_size=3, norm=norm, relu=False, stride=1))
        else:
            layers.append(
                Basic(channel, channel // 2, kernel_size=4, norm=norm, relu=True, stride=2, transpose=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class FOrD_v1(nn.Module):
    def __init__(self, channel, rot_opt=False):
        super(FOrD_v1, self).__init__()

        self.decomp = Basic(channel, channel, kernel_size=1, relu=False, stride=1)

    def forward(self, x):
        x_decomp1 = self.decomp(x)
        x_decomp1_norm = F.normalize(x_decomp1, p=2, dim=1)
        x_decomp2 = x - torch.unsqueeze(torch.sum(x * x_decomp1_norm, dim=1), 1) * x_decomp1_norm
        return x_decomp1, x_decomp2


class TexNet(nn.Module):
    def __init__(self):
        super(TexNet, self).__init__()

        in_channel = 1
        base_channel = 32

        num_res_ENC = 4

        self.Encoder1 = EBlock(in_channel, base_channel, num_res_ENC, first=True)
        self.Encoder2 = EBlock(base_channel, base_channel * 2, num_res_ENC, norm=False)
        self.Encoder3 = EBlock(base_channel * 2, base_channel * 4, num_res_ENC, norm=False)

        self.Convs1_1 = Basic(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1)
        self.Convs1_2 = Basic(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1)

        num_res_DEC = 4

        self.Decoder1_1 = DBlock(base_channel * 4, num_res_DEC, norm=False)
        self.Decoder1_2 = DBlock(base_channel * 2, num_res_DEC, norm=False)
        self.Decoder1_3 = DBlock(base_channel, num_res_DEC, last=True, feature_ensemble=True)
        self.Decoder1_4 = Basic(base_channel, 1, kernel_size=3, relu=False, stride=1)

    def forward(self, x):
        output = list()

        # Common encoder
        x_e1 = self.Encoder1(x)
        x_e2 = self.Encoder2(x_e1)
        x_decomp = self.Encoder3(x_e2)

        # Resultant image reconstruction
        x_decomp1 = self.Decoder1_1(x_decomp)
        x_decomp1 = self.Convs1_1(torch.cat([x_decomp1, x_e2], dim=1))
        x_decomp1 = self.Decoder1_2(x_decomp1)
        x_decomp1 = self.Convs1_2(torch.cat([x_decomp1, x_e1], dim=1))
        x_decomp1 = self.Decoder1_3(x_decomp1)
        x_decomp1 = self.Decoder1_4(x_decomp1)
        return x_decomp1

class CarNet(nn.Module):
    def __init__(self):
        super(CarNet, self).__init__()
        self.TV = utv_model.ADMM(1, 6, 1)
        self.device = torch.device('cuda')
        self.hyp = B.HyPaNet()
    def forward(self, x):
        r = x[:, 0, :, :]
        smoothr = self.TV(r).unsqueeze(1)
        return smoothr

"""
# --------------------------------------------
# (2) Data module, closed-form solution
# It is a trainable-parameter-free module  ^_^
# z_k = D(x_{k-1}, s, k, y, alpha_k)
# some can be pre-calculated
# --------------------------------------------
"""
class DataNet(nn.Module):
    def __init__(self):
        super(DataNet, self).__init__()

    def forward(self, x, FB, FBC, F2B, FBFy, alpha, sf):
        FR = FBFy + torch.rfft(alpha*x, 2, onesided=False)
        x1 = cmul(FB, FR)
        FBR = torch.mean(splits(x1, sf), dim=-1, keepdim=False)
        invW = torch.mean(splits(F2B, sf), dim=-1, keepdim=False)
        invWBR = cdiv(FBR, csum(invW, alpha))
        FCBinvWBR = cmul(FBC, invWBR.repeat(1, 1, sf, sf, 1))
        FX = (FR-FCBinvWBR)/alpha.unsqueeze(-1)
        Xest = torch.irfft(FX, 2, onesided=False)
        return Xest


"""
# --------------------------------------------
# (3) Hyper-parameter module
# --------------------------------------------
"""
class HyPaNet(nn.Module):
    def __init__(self, in_nc=2, out_nc=8, channel=64):
        super(HyPaNet, self).__init__()
        self.mlp = nn.Sequential(
                nn.Conv2d(in_nc, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, out_nc, 1, padding=0, bias=True),
                nn.Softplus())
    def forward(self, x):
        x = self.mlp(x) + 1e-6
        return x


"""
# --------------------------------------------
# main CTDNet
# deep unfolding super-resolution network
# --------------------------------------------
"""
class CTDNet(nn.Module):
    def __init__(self, n_iter=8, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(CTDNet, self).__init__()
        self.d = DataNet()
        self.h = HyPaNet(in_nc=2, out_nc=n_iter*2, channel=h_nc)
        self.n = n_iter
        self.xy = TexNet()
        self.u = CarNet()

    def forward(self, x, k, sf, sigma):
        '''
        x: tensor, NxCxWxH
        k: tensor, Nx(1,3)xwxh
        sf: integer, 1
        sigma: tensor, Nx1x1x1
        '''

        # initialization & pre-calculation
        w, h = x.shape[-2:]
        FB = p2o(k, (w*sf, h*sf))
        FBC = cconj(FB, inplace=False)
        F2B = r2c(cabs2(FB))
        STy = upsample(x, sf=sf)
        FBFy = cmul(FBC, torch.rfft(STy, 2, onesided=False))
        x = nn.functional.interpolate(x, scale_factor=sf, mode='nearest')

        # hyper-parameter
        ab = self.h(torch.cat((sigma, torch.tensor(sf).type_as(sigma).expand_as(sigma)), dim=1))

        # unfolding
        for i in range(self.n):
            x = self.d(x, FB, FBC, F2B, FBFy, ab[:, i:i+1, ...], sf)
            cartoon = self.u(x)
            texture = self.xy(x)
            x = cartoon + texture
        return cartoon, texture, x
