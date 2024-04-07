import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.fft as fft
from torch.nn import functional as F
import tqdm
from typing import Union, List, Dict
import dask.array as da
from aicsimageio import AICSImage
import dask
import warnings

from lsfm_destripe.network import DeStripeModel, GuidedFilterLoss, Loss
from lsfm_destripe.utils import (
    prepare_aux,
    global_correction,
    fusion_perslice,
    destripe_train_params,
)
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
        loss_eps: float = 10,
        qr: float = 0.5,
        resample_ratio: int = 3,
        GF_kernel_size_train: int = 29,
        GF_kernel_size_inference: int = 29,
        hessian_kernel_sigma: float = 1,
        sampling_in_MSEloss: int = 2,
        isotropic_hessian: bool = True,
        lambda_tv: float = 1,
        lambda_hessian: float = 1,
        inc: int = 16,
        n_epochs: int = 300,
        wedge_degree: float = 29,
        n_neighbors: int = 16,
        fast_GF: bool = False,
        require_global_correction: bool = True,
        fusion_GF_kernel_size: int = 49,
        fusion_Gaussian_kernel_size: int = 49,
        device: str = None,
    ):
        self.train_params = {
            "fast_GF": fast_GF,
            "GF_kernel_size_train": GF_kernel_size_train,
            "GF_kernel_size_inference": GF_kernel_size_inference,
            "loss_eps": loss_eps,
            "n_neighbors": n_neighbors,
            "inc": inc,
            "hessian_kernel_sigma": hessian_kernel_sigma,
            "lambda_tv": lambda_tv,
            "lambda_hessian": lambda_hessian,
            "sampling_in_MSEloss": sampling_in_MSEloss,
            "resample_ratio": resample_ratio,
            "isotropic_hessian": isotropic_hessian,
            "n_epochs": n_epochs,
            "wedge_degree": wedge_degree,
            "qr": qr,
            "fusion_GF_kernel_size": fusion_GF_kernel_size,
            "fusion_Gaussian_kernel_size": fusion_Gaussian_kernel_size,
            "require_global_correction": require_global_correction,
        }
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    @staticmethod
    def train_on_one_slice(
        GuidedFilterHRModel,
        sample_params: Dict,
        train_params: Dict,
        X: np.ndarray,
        mask: np.ndarray = None,
        boundary: np.ndarray = None,
        s_: int = 1,
        z: int = 1,
        device: str = "cpu",
    ):
        md = (
            sample_params["md"] if sample_params["is_vertical"] else sample_params["nd"]
        )
        nd = (
            sample_params["nd"] if sample_params["is_vertical"] else sample_params["md"]
        )
        if sample_params["view_num"] > 1:
            assert X.shape[1] == 2, print("input X must have 2 channels.")
            assert isinstance(boundary, np.ndarray), print(
                "dual-view fusion boundary is missing."
            )
            boundary = torch.from_numpy(boundary).to(device)
            kernel = torch.ones(
                1,
                1,
                train_params["fusion_Gaussian_kernel_size"],
                train_params["fusion_Gaussian_kernel_size"],
            ).to(device) / (train_params["fusion_Gaussian_kernel_size"] ** 2)
            dualtarget_numpy = fusion_perslice(
                GuidedFilter(r=train_params["fusion_GF_kernel_size"], eps=1),
                GuidedFilter(r=9, eps=1e-6),
                10 ** X[:, :1, :, :],
                10 ** X[:, 1:, :, :],
                train_params["fusion_Gaussian_kernel_size"],
                kernel,
                boundary,
                device=device,
            )
            dualtarget = torch.from_numpy(
                np.log10(dualtarget_numpy[None, None, :, :])
            ).to(device)
        X, mask = torch.from_numpy(X).to(device), torch.from_numpy(mask).to(device)
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
        mask = F.interpolate(
            mask.float(), (md, nd), align_corners=True, mode="bilinear"
        )
        mask = (mask > 0).float()
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            network = DeStripeModel(
                Angle=sample_params["angle_offset"],
                hier_mask=hier_mask,
                hier_ind=hier_ind,
                NI=NI,
                KS=train_params["GF_kernel_size_train"],
                inc=train_params["inc"],
                m=md,
                n=nd,
                resampleRatio=train_params["resample_ratio"],
                GFr=train_params["fusion_GF_kernel_size"],
                viewnum=sample_params["view_num"],
                device=device,
            ).to(device)
        optimizer = cADAM(network.parameters(), lr=0.01)
        smoothedTarget = GuidedFilterLoss(
            r=train_params["GF_kernel_size_train"], eps=train_params["loss_eps"]
        )(Xd, Xd)
        loss = Loss(train_params, sample_params, device).to(device)
        for epoch in tqdm.tqdm(
            range(train_params["n_epochs"]),
            leave=False if (s_ != z) else True,
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
                mask,
            )  # Xd, X
            epoch_loss.backward()
            optimizer.step()
        with torch.no_grad():
            m, n = X.shape[-2:]
            resultslice = np.zeros(X.shape, dtype=np.float32)
            if not train_params["fast_GF"]:
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
                    Y = fusion_perslice(
                        GuidedFilter(r=train_params["fusion_GF_kernel_size"], eps=1),
                        GuidedFilter(r=9, eps=1e-6),
                        resultslice[:, :1, :, :],
                        resultslice[:, 1:, :, :],
                        train_params["fusion_Gaussian_kernel_size"],
                        kernel,
                        boundary,
                        device=device,
                    )
                else:
                    Y = resultslice[0, 0]
            else:
                for index in range(X.shape[1]):
                    resultslice[:, index : index + 1, :, :] = (
                        10
                        ** GuidedFilterHRModel(
                            Xd[:, index : index + 1, :, :],
                            Y_raw[:, index : index + 1, :, :],
                            X[:, index : index + 1, :, :],
                        )
                        .cpu()
                        .data.numpy()[0, 0]
                    )
                if X.shape[1] > 1:
                    Y = fusion_perslice(
                        GuidedFilter(r=train_params["fusion_GF_kernel_size"], eps=1),
                        GuidedFilter(r=9, eps=1e-6),
                        resultslice[:, :1, :, :],
                        resultslice[:, 1:, :, :],
                        train_params["fusion_Gaussian_kernel_size"],
                        kernel,
                        boundary,
                        device=device,
                    )
                else:
                    Y = resultslice[0, 0]
            return (
                Y,
                resultslice[0] if sample_params["view_num"] > 1 else None,
                dualtarget_numpy if sample_params["view_num"] > 1 else None,
            )

    @staticmethod
    def train_on_full_arr(
        X: Union[np.ndarray, dask.array.core.Array],
        is_vertical: bool,
        angle_offset: List,
        mask: Union[np.ndarray, dask.array.core.Array],
        train_params: Dict = None,
        boundary: np.ndarray = None,
        display: bool = False,
        device: str = "cpu",
    ):
        sample_params = {
            "is_vertical": is_vertical,
            "angle_offset": angle_offset,
        }
        if train_params is None:
            train_params = destripe_train_params()
        else:
            train_params = destripe_train_params(**train_params)
        setup_seed(0)
        z, view_num, m, n = X.shape
        result = np.zeros((z, m, n), dtype=np.uint16)
        mean = np.zeros(z)
        sample_params["view_num"] = view_num
        sample_params["md"], sample_params["nd"] = (
            m // train_params["resample_ratio"] // 2 * 2 + 1,
            n // train_params["resample_ratio"] // 2 * 2 + 1,
        )
        hier_mask_arr, hier_ind_arr, NI_arr = prepare_aux(
            sample_params["md"],
            sample_params["nd"],
            sample_params["is_vertical"],
            sample_params["angle_offset"],
            train_params["wedge_degree"],
            train_params["n_neighbors"],
        )
        train_params["NI"] = NI_arr
        train_params["hier_mask"] = hier_mask_arr
        train_params["hier_ind"] = hier_ind_arr
        if sample_params["view_num"] > 1:
            result_view1, result_view2 = np.zeros((z, m, n), dtype=np.uint16), np.zeros(
                (z, m, n), dtype=np.uint16
            )
            mean_view1, mean_view2 = np.zeros(z), np.zeros(z)
        if train_params["fast_GF"]:
            GuidedFilterHRModel = GuidedFilterHR_fast(
                rx=train_params["GF_kernel_size_inference"],
                ry=0,
                angleList=sample_params["angle_offset"],
                eps=1e-9,
                device=device,
            )
        else:
            GuidedFilterHRModel = GuidedFilterHR(
                rX=[
                    train_params["GF_kernel_size_inference"] * 2 + 1,
                    train_params["GF_kernel_size_inference"],
                ],
                rY=[0, 0],
                m=(m if sample_params["is_vertical"] else n),
                n=(n if sample_params["is_vertical"] else m),
                Angle=sample_params["angle_offset"],
                device=device,
            )
        for i in range(z):
            Ov = np.log10(np.clip(np.asarray(X[i : i + 1]), 1, None))  # (1, v, m, n)
            mask_slice = np.asarray(mask[i : i + 1])[None]
            boundary_slice = (
                boundary[None, None, i : i + 1, :] if boundary is not None else None
            )
            if not sample_params["is_vertical"]:
                Ov, mask_slice = Ov.transpose(0, 1, 3, 2), mask_slice.transpose(
                    0, 1, 3, 2
                )
            Y, resultslice, dualtarget_numpy = DeStripe.train_on_one_slice(
                GuidedFilterHRModel,
                sample_params,
                train_params,
                Ov,
                mask_slice,
                boundary_slice,
                i + 1,
                z,
                device=device,
            )
            if not sample_params["is_vertical"]:
                Y = Y.T
                if sample_params["view_num"] > 1:
                    resultslice = resultslice.transpose(0, 2, 1)
                    dualtarget_numpy = dualtarget_numpy.T
            if display:
                plt.figure(dpi=300)
                ax = plt.subplot(1, 2, 2)
                plt.imshow(Y, vmin=10 ** Ov.min(), vmax=10 ** Ov.max(), cmap="gray")
                ax.set_title("output", fontsize=8, pad=1)
                plt.axis("off")
                ax = plt.subplot(1, 2, 1)
                plt.imshow(
                    dualtarget_numpy if sample_params["view_num"] > 1 else X[i, 0],
                    vmin=10 ** Ov.min(),
                    vmax=10 ** Ov.max(),
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
        if train_params["require_global_correction"] and (z != 1):
            print("global correcting...")
            result = global_correction(mean, result)
            if sample_params["view_num"] > 1:
                result_view1, result_view2 = global_correction(
                    mean_view1, result_view1
                ), global_correction(mean_view2, result_view2)
        print("Done")
        if sample_params["view_num"] == 2:
            return result, result_view1, result_view2
        else:
            return result

    def train(
        self,
        X1: Union[str, np.ndarray, dask.array.core.Array],
        is_vertical: bool,
        angle_offset: List,
        X2: Union[str, np.ndarray, dask.array.core.Array] = None,
        mask: Union[str, np.ndarray, dask.array.core.Array] = None,
        boundary: Union[str, np.ndarray] = None,
        display: bool = False,
    ):
        # read in X
        X1_handle = AICSImage(X1)
        X1_data = X1_handle.get_image_dask_data("ZYX", T=0, C=0)
        if X2 is not None:
            X2_handle = AICSImage(X2)
            X2_data = X2_handle.get_image_dask_data("ZYX", T=0, C=0)
        X = (
            da.stack([X1_data, X2_data], 1)
            if X2 is not None
            else da.stack([X1_data], 1)
        )
        view_num = X.shape[1]
        z, _, m, n = X.shape
        # read in mask
        if mask is None:
            mask_data = np.zeros((z, m, n), dtype=bool)
        else:
            mask_handle = AICSImage(mask)
            mask_data = mask_handle.get_image_dask_data("ZYX", T=0, C=0)
            assert mask_data.shape == (z, m, n), print(
                "mask should be of same shape as input volume(s)."
            )
        # read in dual-result, if applicable
        if view_num > 1:
            assert not isinstance(boundary, type(None)), print(
                "dual-view fusion boundary is missing."
            )
            if isinstance(boundary, str):
                boundary = np.load(boundary)
            assert boundary.shape == (z, n) if is_vertical else (z, m), print(
                "boundary index should be of shape [z_slices, n columns]."
            )
        # training
        out = self.train_on_full_arr(
            X,
            is_vertical,
            angle_offset,
            mask_data,
            self.train_params,
            boundary,
            display=display,
            device=self.device,
        )
        return out
