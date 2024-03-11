import matplotlib.pyplot as plt
import math
import numpy as np
from scipy import ndimage
import torch
import torch.fft as fft
from torch.nn import functional as F
import scipy
import torch.nn as nn
import torchvision
import tqdm
from typing import Union, Tuple, Optional, List, Dict
import dask.array as da
from dask_image.imread import imread

from lsfm_destripe.network import DeStripeModel, GuidedFilterLoss, Loss
from lsfm_destripe.utils import prepare_aux, global_correction, fusion_perslice
from lsfm_destripe.guided_filter_variant import (
    GuidedFilterHR,
    GuidedFilterHR_fast,
    GuidedFilter,
)
from lsfm_destripe.utils_pytorch import cADAM


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


class DeStripe:
    def __init__(
        self,
        is_vertical: bool = True,
        angleOffset: List = [0],
        losseps: float = 10,
        qr: float = 0.5,
        resampleRatio: int = 2,
        KGF: int = 29,
        KGFh: int = 29,
        HKs: float = 0.5,
        sampling_in_MSEloss: int = 2,
        isotropic_hessian: bool = True,
        lambda_tv: float = 1,
        lambda_hessian: float = 1,
        inc: int = 16,
        n_epochs: int = 300,
        deg: float = 29,
        Nneighbors: int = 16,
        fast_GF: bool = False,
        require_global_correction: bool = True,
        GFr: int = 49,
        Gaussianr: int = 49,
    ):
        self.train_params = {
            "fast_GF": fast_GF,
            "KGF": KGF,
            "KGFh": KGFh,
            "losseps": losseps,
            "Nneighbors": Nneighbors,
            "inc": inc,
            "HKs": HKs,
            "lambda_tv": lambda_tv,
            "lambda_hessian": lambda_hessian,
            "sampling": sampling_in_MSEloss,
            "resampleRatio": [resampleRatio, resampleRatio],
            "f": isotropic_hessian,
            "n_epochs": n_epochs,
            "deg": deg,
            "qr": qr,
            "GFr": GFr,
            "Gaussianr": Gaussianr,
            "angleOffset": angleOffset,
        }
        self.sample_params = {
            "require_global_correction": require_global_correction,
            "is_vertical": is_vertical,
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def train_on_one_slice(
        GuidedFilterHRModel,
        sample_params: Dict,
        train_params: Dict,
        X: np.ndarray,
        map: np.ndarray = None,
        dualtarget: np.ndarray = None,
        boundary: np.ndarray = None,
        s_: int = 1,
        z: int = 1,
        device: str = "cuda",
    ):
        md = (
            sample_params["md"] if sample_params["is_vertical"] else sample_params["nd"]
        )
        nd = (
            sample_params["nd"] if sample_params["is_vertical"] else sample_params["md"]
        )
        # put on cuda
        X, map = torch.from_numpy(X).to(device), torch.from_numpy(map).to(device)
        if sample_params["view_num"] > 1:
            assert X.shape[1] == 2, print("input X must have 2 channels.")
            assert isinstance(boundary, np.ndarray), print(
                "dual-view fusion boundary is missing."
            )
            assert isinstance(dualtarget, np.ndarray), print(
                "dual-view fusion result is missing."
            )
            dualtarget = torch.from_numpy(dualtarget).to(device)
            boundary = torch.from_numpy(boundary).to(device)
        # downsample
        Xd = []
        for ind in range(X.shape[1]):
            Xd.append(
                F.interpolate(
                    X[:, ind : ind + 1, :, :],
                    (md, nd),
                    align_corners=True,
                    mode="bilinear",
                )
            )
        Xd = torch.cat(Xd, 1)
        if sample_params["view_num"] > 1:
            dualtargetd = F.interpolate(
                dualtarget, (md, nd), align_corners=True, mode="bilinear"
            )
        map = F.interpolate(map.float(), (md, nd), align_corners=True, mode="bilinear")
        map = (map > 0).float()
        # to Fourier
        Xf = (
            fft.fftshift(fft.fft2(Xd))
            .reshape(1, Xd.shape[1], -1)[0]
            .transpose(1, 0)[: md * nd // 2, :]
        )
        # initialize
        hier_mask, hier_ind, NI = (
            torch.from_numpy(train_params["hier_mask"]).to(device),
            torch.from_numpy(train_params["hier_ind"]).to(device),
            torch.from_numpy(train_params["NI"]).to(device),
        )
        network = DeStripeModel(
            Angle=train_params["angleOffset"],
            hier_mask=hier_mask,
            hier_ind=hier_ind,
            NI=NI,
            KS=train_params["KGF"],
            inc=train_params["inc"],
            m=md,
            n=nd,
            resampleRatio=train_params["resampleRatio"][0],
            GFr=train_params["GFr"],
            viewnum=sample_params["view_num"],
            device=device,
        ).to(device)
        optimizer = cADAM(network.parameters(), lr=0.01)
        smoothedTarget = GuidedFilterLoss(
            r=train_params["KGF"], eps=train_params["losseps"]
        )(Xd, Xd)
        loss = Loss(train_params, sample_params, device).to(device)
        for epoch in tqdm.tqdm(
            range(train_params["n_epochs"]),
            leave=False,
            desc="for {} ({} slices in total): ".format(s_, z),
        ):
            optimizer.zero_grad()
            Y_raw, Y_GNN, Y_LR = network(
                Xd, Xf, Xd if sample_params["view_num"] == 1 else dualtargetd, boundary
            )
            epoch_loss = loss(
                Y_raw,
                Y_GNN,
                Y_LR,
                smoothedTarget,
                Xd if sample_params["view_num"] == 1 else dualtargetd,
                map,
            )  # Xd, X
            epoch_loss.backward()
            optimizer.step()
        with torch.no_grad():
            m, n = X.shape[-2:]
            if train_params["fast_GF"] == False:
                resultslice = np.zeros(X.shape, dtype=np.float32)
                for index in range(X.shape[1]):
                    input2 = X[:, index : index + 1, :, :]
                    input1 = F.interpolate(
                        Y_raw[:, index : index + 1, :, :],
                        (m, n),
                        align_corners=True,
                        mode="bilinear",
                    )
                    input1, input2 = torch.from_numpy(input1.cpu().data.numpy()).to(
                        device
                    ), torch.from_numpy(input2.cpu().data.numpy()).to(device)
                    resultslice[:, index : index + 1, :, :] = (
                        10
                        ** GuidedFilterHRModel(input2, input1, r=train_params["qr"])
                        .cpu()
                        .data.numpy()
                    )
                if X.shape[1] > 1:
                    kernel = torch.ones(
                        1, 1, train_params["Gaussianr"], train_params["Gaussianr"]
                    ).to(device) / (train_params["Gaussianr"] ** 2)
                    Y = fusion_perslice(
                        GuidedFilter(r=train_params["GFr"], eps=1),
                        GuidedFilter(r=9, eps=1e-6),
                        resultslice[:, :1, :, :],
                        resultslice[:, 1:, :, :],
                        train_params["Gaussianr"],
                        kernel,
                        boundary,
                        device=device,
                    )
                else:
                    Y = resultslice[0, 0]
            else:
                Y = (
                    10
                    ** GuidedFilterHRModel(
                        Xd if sample_params["view_num"] == 1 else dualtargetd,
                        Y_GNN,
                        X if sample_params["view_num"] == 1 else dualtarget,
                    )
                    .cpu()
                    .data.numpy()[0, 0]
                )
            return Y, resultslice[0] if sample_params["view_num"] > 1 else None

    @staticmethod
    def train_on_full_arr(
        X,
        sample_params: Dict,
        train_params: Dict,
        map=None,
        dualtarget=None,
        boundary: np.ndarray = None,
        display: bool = False,
        device: str = "cpu",
    ):
        setup_seed(0)
        z, _, m, n = X.shape
        result = np.zeros((z, m, n), dtype=np.uint16)
        mean = np.zeros(z)
        result_view1, result_view2 = None, None
        if sample_params["view_num"] > 1:
            result_view1, result_view2 = np.zeros((z, m, n), dtype=np.uint16), np.zeros(
                (z, m, n), dtype=np.uint16
            )
            mean_view1, mean_view2 = np.zeros(z), np.zeros(z)
        if train_params["fast_GF"]:
            GuidedFilterHRModel = GuidedFilterHR_fast(
                rx=train_params["KGFh"],
                ry=0,
                angleList=train_params["angleOffset"],
                eps=1e-9,
            ).to(device)
        else:
            GuidedFilterHRModel = GuidedFilterHR(
                rX=[train_params["KGFh"] * 2 + 1, train_params["KGFh"]],
                rY=[0, 0],
                m=(
                    sample_params["m"]
                    if sample_params["is_vertical"]
                    else sample_params["n"]
                ),
                n=(
                    sample_params["n"]
                    if sample_params["is_vertical"]
                    else sample_params["m"]
                ),
                Angle=train_params["angleOffset"],
            ).to(device)
        for i in range(z):
            O = np.log10(np.clip(np.asarray(X[i : i + 1]), 1, None))  # (1, v, m, n)
            if sample_params["view_num"] > 1:
                dualtarget_slice = np.log10(
                    np.clip(np.asarray(dualtarget[i : i + 1]), 1, None)
                )[None]
            map_slice = np.asarray(map[i : i + 1])[None]
            boundary_slice = (
                boundary[None, None, i : i + 1, :] if boundary is not None else None
            )
            if not sample_params["is_vertical"]:
                O, map_slice = O.transpose(0, 1, 3, 2), map_slice.transpose(0, 1, 3, 2)
                if sample_params["view_num"] > 1:
                    dualtarget_slice = dualtarget_slice.transpose(0, 1, 3, 2)
            Y, resultslice = DeStripe.train_on_one_slice(
                GuidedFilterHRModel,
                sample_params,
                train_params,
                O,
                map_slice,
                dualtarget_slice if sample_params["view_num"] > 1 else None,
                boundary_slice,
                i + 1,
                z,
                device=device,
            )
            if not sample_params["is_vertical"]:
                Y = Y.T
                if sample_params["view_num"] > 1:
                    resultslice = resultslice.transpose(0, 2, 1)
            if display:
                plt.figure(dpi=300)
                ax = plt.subplot(1, 2, 2)
                plt.imshow(Y, vmin=10 ** O.min(), vmax=10 ** O.max(), cmap="gray")
                ax.set_title("output", fontsize=8, pad=1)
                plt.axis("off")
                ax = plt.subplot(1, 2, 1)
                plt.imshow(
                    dualtarget[i] if sample_params["view_num"] > 1 else X[i, 0],
                    vmin=10 ** O.min(),
                    vmax=10 ** O.max(),
                    cmap="gray",
                )
                ax.set_title("input", fontsize=8, pad=1)
                plt.axis("off")
                plt.show()
            result[i] = np.clip(Y, 0, 65535).astype(np.uint16)
            mean[i] = np.mean(result[i] + 0.1)
            if sample_params["view_num"] > 1:
                result_view1[i] = np.clip(resultslice[:1, :, :], 0, 65535).astype(
                    np.uint16
                )
                result_view2[i] = np.clip(resultslice[1:, :, :], 0, 65535).astype(
                    np.uint16
                )
                mean_view1[i] = np.mean(result_view1[i] + 0.1)
                mean_view2[i] = np.mean(result_view2[i] + 0.1)
        if sample_params["require_global_correction"] and (z != 1):
            print("global correcting...")
            result = global_correction(mean, result)
            if sample_params["view_num"] > 1:
                result_view1, result_view2 = global_correction(
                    mean_view1, result_view1
                ), global_correction(mean_view2, result_view2)
        return result, result_view1, result_view2

    def train(
        self,
        X1: Union[str, np.ndarray],
        X2: Union[str, np.ndarray] = None,
        mask: Union[str, np.ndarray] = None,
        dualX: Union[str, np.ndarray] = None,
        boundary: np.ndarray = None,
        display: bool = False,
    ):
        # read in X
        if isinstance(X1, np.ndarray):
            X1 = X1 if X1.ndim == 3 else X1[None, :, :]
        if isinstance(X2, np.ndarray):
            X2 = X2 if X2.ndim == 3 else X2[None, :, :]
        X1 = imread(X1) if isinstance(X1, str) else X1
        X2 = imread(X2) if isinstance(X2, str) else X2
        X = da.stack([X1, X2], 1) if X2 is not None else da.stack([X1], 1)
        self.sample_params["view_num"] = X.shape[1]
        z, _, m, n = X.shape
        md, nd = (
            m // self.train_params["resampleRatio"][0] // 2 * 2 + 1,
            n // self.train_params["resampleRatio"][1] // 2 * 2 + 1,
        )
        self.sample_params["m"], self.sample_params["n"] = m, n
        self.sample_params["md"], self.sample_params["nd"] = md, nd
        # read in mask
        if mask is None:
            map = np.zeros((z, m, n), dtype=bool)
        else:
            if isinstance(mask, str):
                map = imread(mask)
            elif isinstance(mask, np.ndarray):
                map = mask if mask.ndim == 3 else mask[None, :, :]
            assert map.shape == (z, m, n), print(
                "mask should be of same shape as input volume(s)."
            )
        # read in dual-result, if applicable
        dualtarget = None
        if self.sample_params["view_num"] > 1:
            assert not isinstance(dualX, type(None)), print(
                "dual-view fusion result is missing."
            )
            assert not isinstance(boundary, type(None)), print(
                "dual-view fusion boundary is missing."
            )
            if isinstance(dualX, str):
                dualtarget = imread(dualX)
            elif isinstance(dualX, np.ndarray):
                dualtarget = dualX if dualX.ndim == 3 else dualX[None, :, :]
            assert (
                boundary.shape == (z, n)
                if self.sample_params["is_vertical"]
                else (z, m)
            ), print("boundary index should be of shape [z_slices, n columns].")
        # training
        hier_mask_arr, hier_ind_arr, NI_arr = prepare_aux(
            self.sample_params["md"],
            self.sample_params["nd"],
            self.sample_params["is_vertical"],
            self.train_params["angleOffset"],
            self.train_params["deg"],
            self.train_params["Nneighbors"],
        )
        self.train_params["NI"] = NI_arr
        self.train_params["hier_mask"] = hier_mask_arr
        self.train_params["hier_ind"] = hier_ind_arr
        result, result_view1, result_view2 = self.train_on_full_arr(
            X,
            self.sample_params,
            self.train_params,
            map,
            dualtarget,
            boundary,
            display=display,
            device=self.device,
        )
        return result, result_view1, result_view2
