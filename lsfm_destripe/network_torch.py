import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import scipy
import copy
import torch


class GuidedFilter(nn.Module):
    def __init__(self, rx, ry, r, Angle, m=None, n=None, eps=1e-9):
        super(GuidedFilter, self).__init__()
        Angle = np.rad2deg(np.arctan(r * np.tan(np.deg2rad(Angle))))
        self.pr, self.pc = [], []
        for ind, A in enumerate(Angle):
            lval = np.arange(rx) - rx // 2
            lval = np.round(lval * np.tan(np.deg2rad(A))).astype(np.int32)
            ry = (lval.max() - lval.min()) // 2 * 2 + 1
            i, j = np.meshgrid(np.arange(ry), np.arange(rx))
            i = i - ry // 2
            j = j - rx // 2
            kernel = (i == lval[:, None]).astype(np.float32)[None, None]
            self.register_buffer("kernel" + str(ind), torch.from_numpy(kernel))
            self.pr.append(kernel.shape[-1] // 2)
            self.pc.append(kernel.shape[-2] // 2)

        self.AngleNum = len(Angle)
        self.N = None

    def boxfilter(self, x, k, pc, pr):
        return torch.conv2d(
            F.pad(x, (pr, pr, pc, pc), "constant"), k, stride=1, padding=0
        )

    def __call__(self, X, y, hX, coor):
        if self.N is None:
            XN = torch.ones_like(X)
            self.N = [
                self.boxfilter(
                    XN,
                    getattr(self, "kernel" + str(i)),
                    self.pc[i],
                    self.pr[i],
                )
                for i in range(self.AngleNum)
            ]

        X0 = copy.deepcopy(X)
        for i in range(self.AngleNum):
            b = (
                self.boxfilter(
                    y - X, getattr(self, "kernel" + str(i)), self.pc[i], self.pr[i]
                )
                / self.N[i]
            )
            b = (
                self.boxfilter(
                    b, getattr(self, "kernel" + str(i)), self.pc[i], self.pr[i]
                )
                / self.N[i]
            )
            X = X + b
        hX = (
            F.grid_sample(
                X - X0,
                coor,
                mode="bilinear",
                padding_mode="reflection",
                align_corners=True,
            )
            + hX
        )
        return hX


class complex_relu(nn.Module):
    def __init__(self, inplace=False):
        super(complex_relu, self).__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        return self.relu(x.real) + 1j * self.relu(x.imag)


def Cmplx_Xavier_Init(weight):
    n_in, n_out = weight.shape
    magnitudes = np.ones(weight.shape) / n_in
    phases = np.random.uniform(
        size=(n_in, n_out),
        low=-np.pi,
        high=np.pi,
    )
    return torch.from_numpy(magnitudes * np.exp(1.0j * phases)).to(torch.cfloat)


def CmplxRndUniform(
    bias,
    minval=-0.001,
    maxval=0.001,
):
    real_part = np.random.uniform(
        size=bias.shape,
        low=minval,
        high=maxval,
    )
    imag_part = np.random.uniform(
        size=bias.shape,
        low=minval,
        high=maxval,
    )
    return torch.from_numpy(real_part + 1j * imag_part).to(torch.cfloat)


class gnn(nn.Module):
    def __init__(
        self,
        NI,
        hier_mask,
        hier_ind,
        inc,
    ):
        super(gnn, self).__init__()
        self.register_buffer("hier_mask", hier_mask)
        self.register_buffer("hier_ind", hier_ind)
        self.register_buffer("NI", NI[1:,])
        self.inc = inc
        self.w = nn.Parameter(
            torch.randn(self.NI.shape) + 1j * torch.randn(self.NI.shape)
        )
        self.w.data = Cmplx_Xavier_Init(self.w.data)

    def forward(
        self,
        Xf,
    ):
        Xfcx = torch.sum(
            torch.einsum("knbc,kn->knbc", Xf[self.NI, :], self.w), 0
        )  # (M*N, c)
        Xf_tvx = torch.cat((Xfcx, Xf[self.hier_mask, :]), 0)[self.hier_ind, :].reshape(
            1, -1, 1, self.inc
        )
        return Xf_tvx


class tv_uint(nn.Module):
    def __init__(
        self,
        TVfftx,
        TVffty,
        inverseTVfftx,
        inverseTVffty,
        eigDtD,
        edgeProcess,
        latentProcess,
    ):
        super(tv_uint, self).__init__()
        self.register_buffer("TVfftx", TVfftx)
        self.register_buffer("TVffty", TVffty)
        self.register_buffer("inverseTVfftx", inverseTVfftx)
        self.register_buffer("inverseTVffty", inverseTVffty)
        self.register_buffer("eigDtD", eigDtD)
        self.edgeProcess, self.latentProcess = edgeProcess, latentProcess
        self.complexReLU = complex_relu()

    def forward(self, Xf_tvx, Xf):
        Xf_tvx, Xf_tvy = Xf_tvx * self.TVfftx, Xf * self.TVffty
        X_fourier = []
        for (
            x,
            y,
            inverseTVfftx,
            inverseTVffty,
            eigDtD,
            edgeProcess,
            latentProcess,
        ) in zip(
            Xf_tvx,
            Xf_tvy,
            self.inverseTVfftx,
            self.inverseTVffty,
            self.eigDtD,
            self.edgeProcess,
            self.latentProcess,
        ):
            X_fourier.append(
                self.complexReLU(
                    latentProcess(
                        self.complexReLU(edgeProcess(x) * inverseTVfftx)
                        + self.complexReLU(edgeProcess(y) * inverseTVffty)
                    )
                    / eigDtD
                )
            )
        return torch.cat(X_fourier, -1)


class ResLearning(nn.Module):
    def __init__(self, outc):
        super().__init__()
        self.inc, self.outc = 1, outc
        self.f = nn.Sequential(
            nn.Linear(1, outc).to(torch.cfloat),
            complex_relu(),
            nn.Linear(outc, outc).to(torch.cfloat),
        )
        self.linear = nn.Linear(1, outc).to(torch.cfloat)
        self.complexReLU = complex_relu()
        for m in self.children():
            if isinstance(m, nn.Sequential):
                for n in m.children():
                    if isinstance(n, nn.Linear):
                        n.weight.data = Cmplx_Xavier_Init(n.weight.data)
                        n.bias.data = CmplxRndUniform(n.bias.data)
            else:
                if isinstance(m, nn.Linear):
                    m.weight.data = Cmplx_Xavier_Init(m.weight.data)
                    m.bias.data = CmplxRndUniform(m.bias.data)

    def __call__(self, x):
        inx = self.f(x)
        return self.complexReLU(inx + self.linear(x))


class DeStripeModel_torch(nn.Module):
    def __init__(
        self,
        Angle,
        hier_mask,
        hier_ind,
        NI,
        m_l,
        n_l,
        r,
        inc=16,
    ):
        super(DeStripeModel_torch, self).__init__()
        self.inc = inc
        self.m_l, self.n_l = m_l, n_l
        self.Angle = Angle

        gx_0 = np.fft.fftshift(
            np.fft.fft2(np.array([[1, -1]], dtype=np.float32), (self.m_l, self.n_l))
        )
        gy_0 = np.fft.fftshift(
            np.fft.fft2(np.array([[1], [-1]], dtype=np.float32), (self.m_l, self.n_l))
        )

        TVfftx = []
        inverseTVfftx = []
        TVffty = []
        inverseTVffty = []
        for i, A in enumerate(Angle):
            TVfftx, inverseTVfftx = self.fftnt(
                scipy.ndimage.rotate(
                    gx_0,
                    np.rad2deg(np.arctan(r * np.tan(np.deg2rad(A)))),
                    axes=(-2, -1),
                    reshape=False,
                    order=1,
                    mode="nearest",
                ),
                self.m_l,
                self.n_l,
                TVfftx,
                inverseTVfftx,
            )
            TVffty, inverseTVffty = self.fftn(
                scipy.ndimage.rotate(
                    gy_0,
                    np.rad2deg(np.arctan(r * np.tan(np.deg2rad(A)))),
                    axes=(-2, -1),
                    reshape=False,
                    order=1,
                    mode="nearest",
                ),
                self.m_l,
                self.n_l,
                TVffty,
                inverseTVffty,
            )

        TVfftx = np.concatenate(TVfftx, 0)
        inverseTVfftx = np.concatenate(inverseTVfftx, 0)
        TVffty = np.concatenate(TVffty, 0)
        inverseTVffty = np.concatenate(inverseTVffty, 0)

        eigDtD = np.power(np.abs(TVfftx), 2) + np.power(
            np.abs(TVffty),
            2,
        )

        self.register_buffer(
            "TVfftx", torch.from_numpy(TVfftx)[..., None].to(torch.cfloat)
        )
        self.register_buffer(
            "TVffty", torch.from_numpy(TVffty)[..., None].to(torch.cfloat)
        )
        self.register_buffer(
            "inverseTVfftx", torch.from_numpy(inverseTVfftx)[..., None].to(torch.cfloat)
        )
        self.register_buffer(
            "inverseTVffty", torch.from_numpy(inverseTVffty)[..., None].to(torch.cfloat)
        )
        self.register_buffer(
            "eigDtD", torch.from_numpy(eigDtD)[..., None].to(torch.cfloat)
        )

        self.p = ResLearning(inc)

        self.edgeProcess = []
        for _ in Angle:
            self.edgeProcess.append(
                nn.Sequential(
                    nn.Linear(inc, inc).to(torch.cfloat),
                    complex_relu(),
                    nn.Linear(inc, inc).to(torch.cfloat),
                )
            )
        self.edgeProcess = nn.ModuleList(self.edgeProcess)
        self.latentProcess = []
        for _ in Angle:
            self.latentProcess.append(
                nn.Sequential(
                    nn.Linear(inc, inc).to(torch.cfloat),
                    complex_relu(),
                    nn.Linear(inc, inc).to(torch.cfloat),
                )
            )
        self.latentProcess = nn.ModuleList(self.latentProcess)

        self.merge = nn.Sequential(
            nn.Linear(inc * len(Angle), inc).to(torch.cfloat),
            complex_relu(),
            nn.Linear(inc, inc).to(torch.cfloat),
            complex_relu(),
            nn.Linear(inc, 1).to(torch.cfloat),
        )

        self.gnn = gnn(NI, hier_mask, hier_ind, inc)

        self.tv_uint = tv_uint(
            self.TVfftx,
            self.TVffty,
            self.inverseTVfftx,
            self.inverseTVffty,
            self.eigDtD,
            self.edgeProcess,
            self.latentProcess,
        )

        self.GuidedFilter = GuidedFilter(
            rx=49,
            ry=3,
            r=r,
            m=self.m_l,
            n=self.n_l,
            Angle=Angle,
        )

        self.complexReLU = complex_relu()
        self.alpha = nn.Parameter(torch.ones(1, 1, 1, 1))
        for m in self.children():
            if isinstance(m, nn.Sequential):
                for n in m.children():
                    if isinstance(n, nn.Linear):
                        if n.weight.data.dtype is torch.cfloat:
                            n.weight.data = Cmplx_Xavier_Init(n.weight.data)
                            n.bias.data = CmplxRndUniform(n.bias.data)

    def fftnt(
        self,
        x,
        row,
        col,
        TVfftx,
        inverseTVfftx,
    ):
        y = x.reshape(-1)[: col * row // 2][None, :, None]
        TVfftx.append(y)
        inverseTVfftx.append(np.conj(y))
        return TVfftx, inverseTVfftx

    def fftn(
        self,
        x,
        row,
        col,
        TVffty,
        inverseTVffty,
    ):
        y = x.reshape(-1)[: col * row // 2][None, :, None]
        TVffty.append(y)
        inverseTVffty.append(np.conj(y))
        return TVffty, inverseTVffty

    def fourierResult(
        self,
        z,
        aver,
    ):
        return torch.abs(
            torch.fft.ifft2(
                torch.fft.ifftshift(
                    torch.cat(
                        (z, aver, torch.conj(torch.flip(z, [-2]))),
                        -2,
                    )
                    .reshape(1, self.m_l, -1, 1)
                    .permute(0, 3, 1, 2),
                    dim=(-2, -1),
                )
            )
        )

    def forward(
        self,
        aver,
        Xf,
        target,
        target_hr,
        coor,
    ):
        Xf = self.p(Xf)
        Xf_tvx = self.gnn(Xf)
        X_fourier = self.merge(
            self.tv_uint(
                Xf_tvx,
                Xf,
            )
        )
        outputGNNraw = self.fourierResult(X_fourier[..., 0], aver)
        outputGNNraw = outputGNNraw + self.alpha
        outputLR = self.GuidedFilter(target, outputGNNraw, target_hr, coor)

        return outputGNNraw, outputLR
