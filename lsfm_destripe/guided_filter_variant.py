import torch
import numpy as np
import torchvision
from torch.nn import functional as F
import torch.nn as nn
from scipy import ndimage


class GuidedFilterHR(nn.Module):
    def __init__(self, rX, rY, Angle, m, n, eps=1e-9, device="cpu"):
        super(GuidedFilterHR, self).__init__()
        self.eps = eps
        self.AngleNum = len(Angle)
        self.Angle = Angle
        self.PR, self.PC, self.KERNEL = [], [], []
        for rx, ry in zip(rX, rY):
            self.KERNEL.append(torch.ones((1, 1, rx * 2 + 1, ry * 2 + 1)).to(device))
            self.PR.append(ry)
            self.PC.append(rx)
        self.GaussianKernel = torch.tensor(
            np.ones((rX[1], 1)) * np.ones((1, rX[1])), dtype=torch.float32
        )
        self.GaussianKernel = (self.GaussianKernel / self.GaussianKernel.sum())[
            None, None
        ].to(device)
        self.GaussianKernelpadding = (
            self.GaussianKernel.shape[-2] // 2,
            self.GaussianKernel.shape[-1] // 2,
        )
        self.crop = torchvision.transforms.CenterCrop((m, n))
        self.device = device

    def boxfilter(self, x, weight, k, pc, pr):
        b, c, m, n = x.shape
        img = (
            F.pad(x, (pr, pr, pc, pc))
            .unfold(-2, k.shape[-2], 1)
            .unfold(-2, k.shape[-1], 1)
            .flatten(-2)
        )
        return (img * weight).sum(-1)

    def generateWeight(self, y, k, pc, pr, sigma, device="cpu"):
        b, c, m, n = y.shape
        weight = torch.exp(
            -(
                (
                    F.pad(y, (pr, pr, pc, pc))
                    .unfold(-2, k.shape[-2], 1)
                    .unfold(-2, k.shape[-1], 1)
                    .flatten(-2)
                    - y[:, :, :, :, None]
                )
                ** 2
            )
            / (sigma / 2) ** 2
        )
        d = (torch.linspace(0, k.shape[-2] - 1, k.shape[-2]) - k.shape[-2] // 2)[
            :, None
        ] ** 2 + (torch.linspace(0, k.shape[-1] - 1, k.shape[-1]) - k.shape[-1] // 2)[
            None, :
        ] ** 2
        d = (
            torch.exp(-d.flatten() / (k.numel() / 4) ** 2)[None, None, None, None, :]
            .repeat(b, c, m, n, 1)
            .to(device)
        )
        validPart = (
            F.pad(torch.ones_like(y), (pr, pr, pc, pc))
            .unfold(-2, k.shape[-2], 1)
            .unfold(-2, k.shape[-1], 1)
            .flatten(-2)
        )
        weight[:] = weight * d
        weight[:] = weight * validPart
        weight[:] = weight / weight.sum(-1, keepdim=True)
        return weight

    def forward(self, X, y, r):
        sigma = r * (y.max() - y.min())
        for i, angle in enumerate(self.Angle):
            XX = torchvision.transforms.functional.rotate(X, angle, expand=True)
            yy = torchvision.transforms.functional.rotate(y, angle, expand=True)
            bdetail, Abase, bbase = [], [], []
            XXbase, yybase = torch.conv2d(
                XX, self.GaussianKernel, padding=self.GaussianKernelpadding
            ), torch.conv2d(yy, self.GaussianKernel, padding=self.GaussianKernelpadding)
            XXdetail, yydetail = XX - XXbase, yy - yybase
            list_ = np.arange(XX.shape[-1])
            g, o = 64, 10
            list_ = [list_[i : i + g] for i in range(0, len(list_), g - o)]
            if len(list_[-1]) <= 5:
                list_[-2] = np.arange(list_[-2][0], XX.shape[-1]).tolist()
                del list_[-1]
            for index, i in enumerate(list_):
                Xbase, Xdetail, ybase, ydetail = (
                    XXbase[..., i[0] : i[-1] + 1],
                    XXdetail[..., i[0] : i[-1] + 1],
                    yybase[..., i[0] : i[-1] + 1],
                    yydetail[..., i[0] : i[-1] + 1],
                )
                weightDetail = self.generateWeight(
                    Xdetail,
                    self.KERNEL[0],
                    self.PC[0],
                    self.PR[0],
                    sigma,
                    self.device,
                )
                weightBase = self.generateWeight(
                    Xbase,
                    self.KERNEL[1],
                    self.PC[1],
                    self.PR[1],
                    sigma,
                    self.device,
                )
                bdetail.append(
                    self.boxfilter(
                        ydetail - Xdetail,
                        weightDetail,
                        self.KERNEL[0],
                        self.PC[0],
                        self.PR[0],
                    )[
                        ...,
                        5 if index != 0 else 0 : (
                            -5 if index != len(list_) - 1 else None
                        ),
                    ]
                )
                mean_y, mean_x = self.boxfilter(
                    ybase, weightBase, self.KERNEL[1], self.PC[1], self.PR[1]
                ), self.boxfilter(
                    Xbase, weightBase, self.KERNEL[1], self.PC[1], self.PR[1]
                )
                var_x = (
                    self.boxfilter(
                        Xbase * Xbase,
                        weightBase,
                        self.KERNEL[1],
                        self.PC[1],
                        self.PR[1],
                    )
                    - mean_x * mean_x
                )
                cov_xy = (
                    self.boxfilter(
                        Xbase * ybase,
                        weightBase,
                        self.KERNEL[1],
                        self.PC[1],
                        self.PR[1],
                    )
                    - mean_x * mean_y
                )
                A = cov_xy / (var_x + 1e-6)
                b = mean_y - A * mean_x
                bbase.append(
                    b[
                        ...,
                        5 if index != 0 else 0 : (
                            -5 if index != len(list_) - 1 else None
                        ),
                    ]
                )
                Abase.append(
                    A[
                        ...,
                        5 if index != 0 else 0 : (
                            -5 if index != len(list_) - 1 else None
                        ),
                    ]
                )
            bdetail = torch.cat(bdetail, -1)
            Abase, bbase = torch.cat(Abase, -1), torch.cat(bbase, -1)
            result = Abase * XXbase + bbase + XXdetail + bdetail
            X = self.crop(
                torchvision.transforms.functional.rotate(result, -angle, expand=True)
            )
            y = self.crop(
                torchvision.transforms.functional.rotate(yy, -angle, expand=True)
            )
        return X


class BoxFilter(nn.Module):
    def __init__(self, rx, ry, Angle, device="cpu"):
        super(BoxFilter, self).__init__()
        self.rx, self.ry = rx, ry
        if Angle != 0:
            kernelx = torch.zeros(rx * 4 + 1, rx * 4 + 1)
            kernelx[:, rx * 2] = 1
            kernelx = ndimage.rotate(kernelx, Angle, reshape=False, order=1)[
                rx : 3 * rx + 1, rx : 3 * rx + 1
            ]
            self.kernelx = (
                torch.from_numpy(kernelx[None, None, :, :]).float().to(device)
            )
        else:
            self.kernelx = torch.ones(1, 1, 2 * rx + 1, 2 * ry + 1).float().to(device)
        self.Angle = Angle

    def diff_x(self, input):
        if self.Angle != 0:
            return torch.conv2d(
                F.pad(input, (self.rx, self.rx, self.rx, self.rx), mode="circular"),
                self.kernelx,
            )
        else:
            return torch.conv2d(
                F.pad(input, (self.ry, self.ry, self.rx, self.rx), mode="circular"),
                self.kernelx,
            )

    def forward(self, x):
        return self.diff_x(x)


class GuidedFilterHR_fast(nn.Module):
    def __init__(self, rx, ry, angleList, eps=1e-9, device="cpu"):
        super(GuidedFilterHR_fast, self).__init__()
        self.eps = eps
        self.boxfilter = [
            BoxFilter(rx, ry, Angle=Angle, device=device) for Angle in angleList
        ]
        self.N = None
        self.angleList = angleList
        self.crop = None

    def forward(self, xx, yy, hX):
        if self.crop is None:
            self.crop = torchvision.transforms.CenterCrop(xx.size()[-2:])
        AList, bList = [], []
        for i, Angle in enumerate(self.angleList):
            x = torchvision.transforms.functional.rotate(xx, Angle, expand=True)
            y = torchvision.transforms.functional.rotate(yy, Angle, expand=True)
            h_x, w_x = x.size()[-2:]
            self.N = self.boxfilter[i](
                x.data.new().resize_((1, 1, h_x, w_x)).fill_(1.0)
            )
            mean_x = self.boxfilter[i](x) / self.N
            var_x = self.boxfilter[i](x * x) / self.N - mean_x * mean_x
            mean_y = self.boxfilter[i](y) / self.N
            cov_xy = self.boxfilter[i](x * y) / self.N - mean_x * mean_y
            A = cov_xy / (var_x + self.eps)
            b = mean_y - A * mean_x
            A, b = (self.boxfilter[i](A) / self.N), self.boxfilter[i](b) / self.N
            result = A * x + b
            AList.append(A)
            bList.append(b)
            xx = self.crop(
                torchvision.transforms.functional.rotate(result, -Angle, expand=True)
            )
            yy = self.crop(
                torchvision.transforms.functional.rotate(yy, -Angle, expand=True)
            )
        with torch.no_grad():
            h_hrx, w_hrx = hX.size()[-2:]
            cropH = torchvision.transforms.CenterCrop((h_hrx, w_hrx))
            for i, (Angle, A, b) in enumerate(zip(self.angleList, AList, bList)):
                hXX = torchvision.transforms.functional.rotate(hX, Angle, expand=True)
                hA = F.interpolate(
                    A, hXX.shape[-2:], mode="bilinear", align_corners=True
                )
                hb = F.interpolate(
                    b, hXX.shape[-2:], mode="bilinear", align_corners=True
                )
                result = hA * hXX + hb
                hX = cropH(
                    torchvision.transforms.functional.rotate(
                        result, -Angle, expand=True
                    )
                )
                return hX


class BoxFilter2:
    def __init__(self, r):
        self.r = r

    def diff_x(self, input, r):
        left = input[:, :, r : 2 * r + 1]
        middle = input[:, :, 2 * r + 1 :] - input[:, :, : -2 * r - 1]
        right = input[:, :, -1:] - input[:, :, -2 * r - 1 : -r - 1]
        output = torch.cat([left, middle, right], dim=2)
        return output

    def diff_y(self, input, r):
        left = input[:, :, :, r : 2 * r + 1]
        middle = input[:, :, :, 2 * r + 1 :] - input[:, :, :, : -2 * r - 1]
        right = input[:, :, :, -1:] - input[:, :, :, -2 * r - 1 : -r - 1]
        output = torch.cat([left, middle, right], dim=3)
        return output

    def __call__(self, x):
        return self.diff_y(self.diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)


class GuidedFilter:
    def __init__(self, r, eps=1e-8):
        self.r, self.eps = r, eps
        self.boxfilter = BoxFilter2(r)

    def __call__(self, x, y):
        x, y = 0.001 * x, 0.001 * y
        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()
        N = self.boxfilter(x.data.new().resize_((1, 1, h_x, w_x)).fill_(1.0))
        mean_x = self.boxfilter(x) / N
        mean_y = self.boxfilter(y) / N
        cov_xy = self.boxfilter(x * y) / N - mean_x * mean_y
        var_x = self.boxfilter(x * x) / N - mean_x * mean_x
        A = cov_xy / (var_x + self.eps)
        b = mean_y - A * mean_x
        mean_A = self.boxfilter(A) / N
        mean_b = self.boxfilter(b) / N
        return (mean_A * x + mean_b) / 0.001
