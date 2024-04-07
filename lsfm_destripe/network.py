import math
import numpy as np
from scipy import ndimage
import torch
import torch.fft as fft
from torch.nn import functional as F
import torch.nn as nn
import scipy


def Cmplx_Xavier_Init(weight):
    n_in, n_out = weight.shape
    sigma = 1 / np.sqrt(n_in + n_out)
    magnitudes = np.random.rayleigh(scale=sigma, size=(n_in, n_out))
    phases = np.random.uniform(size=(n_in, n_out), low=-np.pi, high=np.pi)
    return torch.from_numpy(magnitudes * np.exp(1.0j * phases)).to(torch.cfloat)


def CmplxRndUniform(bias, minval, maxval):
    real_part = np.random.uniform(size=bias.shape, low=minval, high=maxval)
    imag_part = np.random.uniform(size=bias.shape, low=minval, high=maxval)
    return torch.from_numpy(real_part + 1j * imag_part).to(torch.cfloat)


class complexReLU(nn.Module):
    def __init__(self, inplace=False):
        super(complexReLU, self).__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        return self.relu(x.real) + 1j * self.relu(x.imag)


class ResLearning(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.inc, self.outc = inc, outc
        self.f = nn.Sequential(
            nn.Linear(inc, outc).to(torch.cfloat),
            complexReLU(),
            nn.Linear(outc, outc).to(torch.cfloat),
        )
        self.linear = nn.Linear(inc, outc).to(torch.cfloat)
        self.complexReLU = complexReLU()
        for m in self.children():
            if isinstance(m, nn.Sequential):
                for n in m.children():
                    if isinstance(n, nn.Linear):
                        n.weight.data = Cmplx_Xavier_Init(n.weight.data)
                        n.bias.data = CmplxRndUniform(n.bias.data, -0.001, 0.001)
            else:
                if isinstance(m, nn.Linear):
                    m.weight.data = Cmplx_Xavier_Init(m.weight.data)
                    m.bias.data = CmplxRndUniform(m.bias.data, -0.001, 0.001)

    def __call__(self, x):
        inx = self.f(x)
        return self.complexReLU(inx + self.linear(x))


class GuidedFilter(nn.Module):
    def __init__(self, rx, ry, Angle, m, n, device="cpu", eps=1e-9):
        super().__init__()
        self.eps = eps
        self.AngleNum = len(Angle)
        self.Angle = Angle
        kernelL = []
        for A in Angle:
            kernel = np.zeros((rx * 4 + 1, rx * 4 + 1), dtype=np.float32)
            if ry == 0:
                kernel[:, rx * 2] = 1
            else:
                kernel[:, rx * 2 - ry : rx * 2 + ry + 1] = 1
            kernel = ndimage.rotate(kernel, A, reshape=False, order=2)[
                rx : 3 * rx + 1, rx : 3 * rx + 1
            ]
            r, c = sum(kernel.sum(1) != 0) // 2 * 2, sum(kernel.sum(0) != 0) // 2 * 2
            kernelL.append(
                kernel[rx - r // 2 : rx + r // 2 + 1, rx - c // 2 : rx + c // 2 + 1][
                    None, None
                ]
            )
        c, r = max([k.shape[-2] for k in kernelL]), max([k.shape[-1] for k in kernelL])
        self.kernel = np.zeros((len(Angle), 1, c, r))
        for i, k in enumerate(kernelL):
            self.kernel[
                i,
                :,
                (c - k.shape[-2])
                // 2 : (
                    -(c - k.shape[-2]) // 2 if -(c - k.shape[-2]) // 2 != 0 else None
                ),
                (r - k.shape[-1])
                // 2 : (
                    -(r - k.shape[-1]) // 2 if -(r - k.shape[-1]) // 2 != 0 else None
                ),
            ] = k
        self.kernel = torch.from_numpy(self.kernel).to(torch.float).to(device)
        self.pr, self.pc = self.kernel.shape[-1] // 2, self.kernel.shape[-2] // 2
        XN = torch.ones((1, 1, m, n)).to(device)
        self.N = [
            self.boxfilter(XN, self.kernel[i : i + 1, ...])
            for i in range(self.AngleNum)
        ]

    def boxfilter(self, x, k):
        return torch.conv2d(F.pad(x, (self.pr, self.pr, self.pc, self.pc)), k, groups=1)

    def __call__(self, X, y):
        for i in range(self.AngleNum):
            mean_y, mean_x = (
                self.boxfilter(y, self.kernel[i : i + 1, ...]) / self.N[i],
                self.boxfilter(X, self.kernel[i : i + 1, ...]) / self.N[i],
            )
            b = mean_y - mean_x
            X = X + b
        return X


class dual_view_fusion:
    def __init__(self, r, m, n, resampleRatio, eps=1, device="cpu"):
        self.r = r
        self.mask = torch.arange(m)[None, None, :, None].to(device)
        self.m, self.n = m, n
        self.eps = eps
        self.resampleRatio = resampleRatio

    def diff_x(self, input, r):
        left = input[:, :, r : 2 * r + 1]
        middle = input[:, :, 2 * r + 1 :] - input[:, :, : -2 * r - 1]
        right = input[:, :, -1:] - input[:, :, -2 * r - 1 : -r - 1]
        output = torch.cat([left, middle, right], 2)
        return output

    def diff_y(self, input, r):
        left = input[:, :, :, r : 2 * r + 1]
        middle = input[:, :, :, 2 * r + 1 :] - input[:, :, :, : -2 * r - 1]
        right = input[:, :, :, -1:] - input[:, :, :, -2 * r - 1 : -r - 1]
        output = torch.cat([left, middle, right], 3)
        return output

    def boxfilter(self, input):
        return self.diff_y(self.diff_x(input.cumsum(2), self.r).cumsum(3), self.r)

    def guidedfilter(self, x, y):
        N = self.boxfilter(torch.ones_like(x))
        mean_x, mean_y = self.boxfilter(x) / N, self.boxfilter(y) / N
        cov_xy = self.boxfilter(x * y) / N - mean_x * mean_y
        var_x = self.boxfilter(x * x) / N - mean_x * mean_x
        A = cov_xy / torch.clip(var_x, self.eps, None)
        b = mean_y - A * mean_x
        A, b = self.boxfilter(A) / N, self.boxfilter(b) / N
        return A * x + b

    def __call__(self, x, boundary):
        boundary = (
            F.interpolate(
                boundary, size=(1, self.n), mode="bilinear", align_corners=True
            )
            / self.resampleRatio
        )
        topSlice, bottomSlice = torch.split(x, split_size_or_sections=[1, 1], dim=1)
        mask0, mask1 = self.mask > boundary, self.mask <= boundary
        result0, result1 = self.guidedfilter(bottomSlice, mask0), self.guidedfilter(
            topSlice, mask1
        )
        t = result0 + result1 + 1e-3
        result0, result1 = result0 / t, result1 / t
        return result0 * bottomSlice + result1 * topSlice


class identical_func:
    def __init__(
        self,
    ):
        pass

    def __call__(self, x, boundary):
        return x


class DeStripeModel(nn.Module):
    def __init__(
        self,
        Angle,
        hier_mask,
        hier_ind,
        NI,
        m,
        n,
        resampleRatio,
        KS,
        inc=16,
        GFr=49,
        viewnum=1,
        device="cpu",
    ):
        super(DeStripeModel, self).__init__()
        self.hier_mask = hier_mask
        self.hier_ind = hier_ind
        self.NI = NI
        self.Angle = Angle
        self.m, self.n = m, n
        gx, gy = self.rotatableKernel(Wsize=1, sigma=1)
        if len(Angle) > 1:
            self.TVfftx, self.inverseTVfftx, self.TVffty, self.inverseTVffty = (
                [],
                [],
                [],
                [],
            )
            for i, A in enumerate(Angle):
                self.fftnt(
                    math.cos(-A / 180 * math.pi) * gx
                    + math.sin(-A / 180 * math.pi) * gy,
                    m,
                    n,
                )
                self.fftn(
                    math.cos(-A / 180 * math.pi + math.pi / 2) * gx
                    + math.sin(-A / 180 * math.pi + math.pi / 2) * gy,
                    m,
                    n,
                )
            self.TVfftx, self.inverseTVfftx = np.concatenate(
                self.TVfftx, 0
            ), np.concatenate(self.inverseTVfftx, 0)
            self.TVffty, self.inverseTVffty = np.concatenate(
                self.TVffty, 0
            ), np.concatenate(self.inverseTVffty, 0)
        else:
            if Angle[0] != 0:
                self.TVfftx, self.inverseTVfftx = self.fftnt(
                    math.cos(-Angle[0] / 180 * math.pi) * gx
                    + math.sin(-Angle[0] / 180 * math.pi) * gy,
                    m,
                    n,
                )
                self.TVffty, self.inverseTVffty = self.fftn(
                    math.cos(-Angle[0] / 180 * math.pi + math.pi / 2) * gx
                    + math.sin(-Angle[0] / 180 * math.pi + math.pi / 2) * gy,
                    m,
                    n,
                )
            else:
                self.TVfftx, self.inverseTVfftx = self.fftnt(
                    np.array([[1, -1]], dtype=np.float32), m, n
                )
                self.TVffty, self.inverseTVffty = self.fftn(
                    np.array([[1], [-1]], dtype=np.float32), m, n
                )
        self.TVfftx = torch.from_numpy(self.TVfftx).to(torch.cfloat).to(device)
        self.TVffty = torch.from_numpy(self.TVffty).to(torch.cfloat).to(device)
        self.inverseTVfftx = (
            torch.from_numpy(self.inverseTVfftx).to(torch.cfloat).to(device)
        )
        self.inverseTVffty = (
            torch.from_numpy(self.inverseTVffty).to(torch.cfloat).to(device)
        )
        self.eigDtD = torch.div(
            1,
            torch.pow(torch.abs(self.TVfftx), 2) + torch.pow(torch.abs(self.TVffty), 2),
        )
        self.p = ResLearning(viewnum, inc)
        self.edgeProcess = nn.Sequential(
            nn.Linear(inc, inc).to(torch.cfloat),
            complexReLU(),
            nn.Linear(inc, inc).to(torch.cfloat),
        )
        self.latentProcess = nn.Sequential(
            nn.Linear(inc, inc).to(torch.cfloat),
            complexReLU(),
            nn.Linear(inc, inc).to(torch.cfloat),
        )
        self.merge = nn.Sequential(
            nn.Linear(inc * len(Angle), inc).to(torch.cfloat),
            complexReLU(),
            nn.Linear(inc, inc).to(torch.cfloat),
            complexReLU(),
            nn.Linear(inc, viewnum).to(torch.cfloat),
        )
        self.base = nn.Sequential(
            nn.Linear(1, inc),
            nn.ReLU(inplace=True),
            nn.Linear(inc, inc),
            nn.ReLU(inplace=True),
            nn.Linear(inc, viewnum),
        )
        self.viewnum = viewnum
        self.GuidedFilter = GuidedFilter(
            rx=KS, ry=0, m=m, n=n, Angle=Angle, device=device
        )
        self.w = nn.Parameter(torch.rand(NI.shape) + 1j * torch.rand(NI.shape))
        self.w.data = Cmplx_Xavier_Init(self.w.data)
        self.complexReLU = complexReLU()
        self.ainput = torch.ones(1, 1).to(device)
        self.inc = inc
        self.fuse = (
            dual_view_fusion(GFr, m, n, resampleRatio, device=device)
            if viewnum == 2
            else identical_func()
        )
        for m in self.children():
            if isinstance(m, nn.Sequential):
                for n in m.children():
                    if isinstance(n, nn.Linear):
                        if n.weight.data.dtype is torch.cfloat:
                            n.weight.data = Cmplx_Xavier_Init(n.weight.data)
                            n.bias.data = CmplxRndUniform(n.bias.data, -0.001, 0.001)

    def rotatableKernel(self, Wsize, sigma):
        k = np.linspace(-Wsize, Wsize, 2 * Wsize + 1)[None, :]
        g, gp = np.exp(-(k**2) / (2 * sigma**2)), -(k / sigma) * np.exp(
            -(k**2) / (2 * sigma**2)
        )
        return g.T * gp, gp.T * g

    def fftnt(self, x, row, col):
        y = np.fft.fftshift(np.fft.fft2(x, s=(row, col))).reshape(-1)[: col * row // 2][
            None, :, None
        ]
        if len(self.Angle) > 1:
            self.TVfftx.append(y)
            self.inverseTVfftx.append(np.conj(y))
        else:
            return y, np.conj(y)

    def fftn(self, x, row, col):
        y = np.fft.fftshift(np.fft.fft2(x, s=(row, col))).reshape(-1)[: col * row // 2][
            None, :, None
        ]
        if len(self.Angle) > 1:
            self.TVffty.append(y)
            self.inverseTVffty.append(np.conj(y))
        else:
            return y, np.conj(y)

    def fourierResult(self, z, mean):
        return torch.abs(
            fft.ifft2(
                fft.ifftshift(
                    torch.cat(
                        (
                            z,
                            self.base(self.ainput) * mean,
                            torch.conj(torch.flip(z, [0])),
                        ),
                        0,
                    )
                    .reshape(1, self.m, -1, self.viewnum)
                    .permute(0, 3, 1, 2)
                )
            )
        )

    def forward(self, Xd, Xf, target, boundary):
        aver = Xd.sum(dim=(2, 3)) + 1j * 0
        Xf = self.p(Xf)
        Xfcx = torch.sum(torch.einsum("knc,kn->knc", Xf[self.NI, :], self.w), 0)
        Xf_tvx = torch.cat((Xfcx, Xf[self.hier_mask, :]), 0)[self.hier_ind, :].reshape(
            len(self.Angle), -1, self.inc
        )
        Xf_tvx, Xf_tvy = Xf_tvx * self.TVfftx, Xf * self.TVffty
        X_fourier = []
        for x, y, inverseTVfftx, inverseTVffty, eigDtD in zip(
            Xf_tvx, Xf_tvy, self.inverseTVfftx, self.inverseTVffty, self.eigDtD
        ):
            X_fourier.append(
                self.complexReLU(
                    self.latentProcess(
                        self.complexReLU(self.edgeProcess(x) * inverseTVfftx)
                        + self.complexReLU(self.edgeProcess(y) * inverseTVffty)
                    )
                    * eigDtD
                )
            )
        X_fourier = self.merge(torch.cat(X_fourier, -1))
        outputGNNraw = self.fourierResult(X_fourier, aver)
        outputGNN = self.fuse(outputGNNraw, boundary)
        outputLR = self.GuidedFilter(target, outputGNN)
        return outputGNNraw, outputGNN, outputLR


class GuidedFilterLoss(nn.Module):
    def __init__(self, r, eps=1e-9):
        super(GuidedFilterLoss, self).__init__()
        self.r, self.eps = r, eps

    def diff_x(self, input, r):
        return input[:, :, 2 * r + 1 :, :] - input[:, :, : -2 * r - 1, :]

    def diff_y(self, input, r):
        return input[:, :, :, 2 * r + 1 :] - input[:, :, :, : -2 * r - 1]

    def boxfilter(self, input):
        return self.diff_x(
            self.diff_y(
                F.pad(
                    input, (self.r + 1, self.r, self.r + 1, self.r), mode="constant"
                ).cumsum(3),
                self.r,
            ).cumsum(2),
            self.r,
        )

    def forward(self, x, y):
        N = self.boxfilter(torch.ones_like(x))
        mean_x, mean_y = self.boxfilter(x) / N, self.boxfilter(y) / N
        cov_xy = self.boxfilter(x * y) / N - mean_x * mean_y
        var_x = self.boxfilter(x * x) / N - mean_x * mean_x
        A = cov_xy / torch.clip(var_x, self.eps, None)
        b = mean_y - A * mean_x
        A, b = self.boxfilter(A) / N, self.boxfilter(b) / N
        return A * x + b


class Loss(nn.Module):
    def __init__(self, train_params, shape_params, device):
        super(Loss, self).__init__()
        self.lambda_tv = train_params["lambda_tv"]
        self.lambda_hessian = train_params["lambda_hessian"]
        self.angleOffset = shape_params["angle_offset"]
        self.sampling = train_params["sampling_in_MSEloss"]
        self.f = train_params["isotropic_hessian"]
        self.Dy = torch.from_numpy(
            np.array([[1], [-1]], dtype=np.float32)[None, None]
        ).to(device)
        self.Dx = torch.from_numpy(
            np.array([[1, -1]], dtype=np.float32)[None, None]
        ).to(device)
        if train_params["hessian_kernel_sigma"] > 0.5:
            self.DGaussxx, self.DGaussyy, self.DGaussxy = self.generateHessianKernel(
                train_params["hessian_kernel_sigma"]
            )
        else:
            self.DGaussxx, self.DGaussyy, self.DGaussxy = self.generateHessianKernel2(
                train_params["hessian_kernel_sigma"], shape_params
            )
        self.DGaussxx, self.DGaussyy, self.DGaussxy = (
            self.DGaussxx.to(device),
            self.DGaussyy.to(device),
            self.DGaussxy.to(device),
        )
        self.GuidedFilterLoss = GuidedFilterLoss(
            r=train_params["GF_kernel_size_train"], eps=train_params["loss_eps"]
        )

    def generateHessianKernel2(self, Sigma, shape_params):
        Wsize = math.ceil(3 * Sigma)
        KernelSize = 2 * (2 * Wsize + 1) - 1
        gx, gy = self.rotatableKernel(Wsize, Sigma)
        md = shape_params["md"] if shape_params["is_vertical"] else shape_params["nd"]
        nd = shape_params["nd"] if shape_params["is_vertical"] else shape_params["md"]
        gxFFT2, gyFFT2 = fft.fft2(gx, s=(md, nd)), fft.fft2(gy, s=(md, nd))
        DGaussxx = torch.zeros(len(self.angleOffset), 1, KernelSize, KernelSize)
        DGaussxy = torch.zeros(len(self.angleOffset), 1, KernelSize, KernelSize)
        DGaussyy = torch.zeros(len(self.angleOffset), 1, KernelSize, KernelSize)
        for i, A in enumerate(self.angleOffset):
            a, b, c, d = (
                math.cos(-A / 180 * math.pi),
                math.sin(-A / 180 * math.pi),
                math.cos(-A / 180 * math.pi + math.pi / 2),
                math.sin(-A / 180 * math.pi + math.pi / 2),
            )
            DGaussxx[i] = (
                fft.ifft2((a * gxFFT2 + b * gyFFT2) * (a * gxFFT2 + b * gyFFT2))
                .real[None, None, :KernelSize, :KernelSize]
                .float()
            )
            DGaussyy[i] = (
                fft.ifft2((c * gxFFT2 + d * gyFFT2) * (c * gxFFT2 + d * gyFFT2))
                .real[None, None, :KernelSize, :KernelSize]
                .float()
            )
            DGaussxy[i] = (
                fft.ifft2((c * gxFFT2 + d * gyFFT2) * (a * gxFFT2 + b * gyFFT2))
                .real[None, None, :KernelSize, :KernelSize]
                .float()
            )
        return (
            torch.from_numpy(DGaussxx.data.numpy()),
            torch.from_numpy(DGaussyy.data.numpy()),
            torch.from_numpy(DGaussxy.data.numpy()),
        )

    def rotatableKernel(self, Wsize, sigma):
        k = torch.linspace(-Wsize, Wsize, 2 * Wsize + 1)[None, :]
        g = torch.exp(-(k**2) / (2 * sigma**2))
        gp = -(k / sigma) * torch.exp(-(k**2) / (2 * sigma**2))
        return g.T * gp, gp.T * g

    def generateHessianKernel(self, Sigma):
        tmp = np.linspace(
            -1 * np.ceil(Sigma * 6), np.ceil(Sigma * 6), int(np.ceil(Sigma * 6) * 2 + 1)
        )
        X, Y = np.meshgrid(tmp, tmp, indexing="ij")
        DGaussxx = torch.from_numpy(
            1
            / (2 * math.pi * Sigma**4)
            * (X**2 / Sigma**2 - 1)
            * np.exp(-(X**2 + Y**2) / (2 * Sigma**2))
        )[None, None, :, :]
        DGaussxy = torch.from_numpy(
            1
            / (2 * math.pi * Sigma**6)
            * (X * Y)
            * np.exp(-(X**2 + Y**2) / (2 * Sigma**2))
        )[None, None, :, :]
        DGaussyy = DGaussxx.transpose(3, 2)
        DGaussxx, DGaussxy, DGaussyy = (
            DGaussxx.float().data.numpy(),
            DGaussxy.float().data.numpy(),
            DGaussyy.float().data.numpy(),
        )
        Gaussxx, Gaussxy, Gaussyy = [], [], []
        for A in self.angleOffset:
            Gaussxx.append(
                scipy.ndimage.rotate(DGaussxx, A, axes=(-2, -1), reshape=False)
            )
            Gaussyy.append(
                scipy.ndimage.rotate(DGaussyy, A, axes=(-2, -1), reshape=False)
            )
            Gaussxy.append(
                scipy.ndimage.rotate(DGaussxy, A, axes=(-2, -1), reshape=False)
            )
        Gaussxx, Gaussyy, Gaussxy = (
            np.concatenate(Gaussxx, 0),
            np.concatenate(Gaussyy, 0),
            np.concatenate(Gaussxy, 0),
        )
        return (
            torch.from_numpy(Gaussyy),
            torch.from_numpy(Gaussxx),
            torch.from_numpy(Gaussxy),
        )

    def TotalVariation(self, x, target):
        return (
            torch.abs(torch.conv2d(x, self.Dx)).sum()
            + torch.abs(torch.conv2d(x - target, self.Dy)).sum()
        )

    def HessianRegularizationLoss(self, x, target):
        return (
            torch.abs(torch.conv2d(x, self.DGaussxx)).sum()
            + torch.abs(torch.conv2d(x - target, self.DGaussyy)).sum()
            + (
                torch.abs(torch.conv2d(x - target, self.DGaussxy))
                * (2 if self.f else 1)
            ).sum()
        )

    def forward(self, outputGNNraw, outputGNN, outputLR, smoothedTarget, targets, map):
        mse = torch.sum(
            torch.abs(
                smoothedTarget - self.GuidedFilterLoss(outputGNNraw, outputGNNraw)
            )
        ) + torch.sum(
            (torch.abs(targets - outputGNN) * map)[
                :, :, :: self.sampling, :: self.sampling
            ]
        )
        tv = 1 * self.TotalVariation(outputGNN, targets) + 1 * self.TotalVariation(
            outputLR, targets
        )
        hessian = 1 * self.HessianRegularizationLoss(
            outputGNN, targets
        ) + 1 * self.HessianRegularizationLoss(outputLR, targets)
        return mse + self.lambda_tv * tv + self.lambda_hessian * hessian
