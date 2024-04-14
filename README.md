# LSFM DeStripe PyTorch

[![Build Status](https://github.com/peng-lab/lsfm_destripe/workflows/Build%20Main/badge.svg)](https://github.com/peng-lab/lsfm_destripe/actions)
[![Documentation](https://github.com/peng-lab/lsfm_destripe/workflows/Documentation/badge.svg)](https://peng-lab.github.io/lsfm_destripe/)
[![Code Coverage](https://codecov.io/gh/peng-lab/lsfm_destripe/branch/main/graph/badge.svg)](https://codecov.io/gh/peng-lab/lsfm_destripe)

A PyTorch implementation of LSFM DeStripe method

(add a short description, to be updated by Yu)

---

## Quick Start
### Use as Python API
(1) Provide a filename, run slice by slice (suitable for extremely large file)
suppose we have a to-be-processed volume saved in `data_path` with vertical stripe in it
```python
from lsfm_destripe import DeStripe

exe = DeStripe()  ###to run with default parameters for training
out = exe.train(
    X1=data_path,
    is_vertical=True,
    angle_offset=[0],
    mask=mask_path,
)
```
where the volume in `mask_path` is to specifically indicate structures that are to be reserved after processing, left out can run prue DeStripe without constraint, `is_vertical` is to define whether the stirpes, or say the direction of light sheet propagation is along vertical or horizona. In practice, the stripe maight not be strictly vertical/horizontal, thus the angle offset in degree can be defined in `angle_offset`. Moreover, for multi-directional LSFM, for example Ultramicroscope II, Lavision Biotec. whose outputs exhibit stripes in 3 directions, -10 degrees, 0 degress, and 10 degrees, DeStripe can also remove all the stripes at the same time by runing:
```python
out = exe.train(
    X1=data_path,
    is_vertical=True,
    angle_offset=[-10, 0, 10],
    mask=mask_path,
)
```
Alternatively, DeStripe can be initialized with user-defined train parameters, a full list of input arguments in `__init__` is here:
```
loss_eps: float = 10
qr: float = 0.5
resample_ratio: int = 3
GF_kernel_size_train: int = 29
GF_kernel_size_inference: int = 29
hessian_kernel_sigma: float = 1
sampling_in_MSEloss: int = 2
isotropic_hessian: bool = True
lambda_tv: float = 1
lambda_hessian: float = 1
inc: int = 16
n_epochs: int = 300
wedge_degree: float = 29
n_neighbors: int = 16
fast_GF: bool = False
require_global_correction: bool = True
fusion_GF_kernel_size: int = 49
fusion_Gaussian_kernel_size: int = 49
device: str = None
```

(2) Provide a array, i.e., the image to be processed, and necessary parameters (more suitable for small data or for use in napari)
suppose we have a to-be-processed volume `img_arr` that has been read in as a np.ndarray or dask.array.core.Array and has vertical stripes in it
```python
from lsfm_destripe import DeStripe

###run with default training params
out = DeStripe.train_on_full_arr(
    X=img_arr,
    is_vertical=True,
    angle_offset=[0],
    mask=mask_arr,
    device="cuda",
)
```
where `img_arr` has a size of $S \times 1 \times M \times N$, `mask_arr` has  a size of $S \times M \times N$ with element 1 and 0 giving reserve or not respectively, `device` by default is "cpu".
Also, customized training can be done by wrapping the training params that you'd like to change as a Dict.:
```python
from lsfm_destripe import DeStripe

out = DeStripe.train_on_full_arr(
    X=img_arr,
    is_vertical=True,
    angle_offset=[0],
    mask=mask_arr,
    device="cuda",
    train_params = {"resample_ratio": 3},
)
```
### Run from command line for batch processing
suppose we have a to-be-processed volume saved in /path/to/my/image.tiff and it has vertical stripes, and we'd like to save the result from DeStripe as /path/to/save/result.tif
```bash
destripe --X1_path /path/to/my/image.tiff \
         --is_vertical True \
         --angle_offset 0 \
         --save_path /path/to/save/result.tif
```
in addition, all the training parameters can be changed in command line:
```bash
destripe --X1_path /path/to/my/image.tiff \
         --is_vertical True \
         --angle_offset 0,-10,10 \
         --save_path /path/to/save/result.tif
```
a full list of changeable args:
```
usage: run_destripe --X1_path
                    --save_path
                    --is_vertical
                    --angle_offset
                    [--loss_eps 10]
                    [--qr 0.5]
                    [--resample_ratio 3]
                    [--GF_kernel_size_train 29]
                    [--GF_kernel_size_inference 29]
                    [--hessian_kernel_sigma 1]
                    [--sampling_in_MSEloss 2]
                    [--isotropic_hessian "True"]
                    [--lambda_tv 1]
                    [--lambda_hessian 1]
                    [--inc 16]
                    [--n_epochs 300]
                    [--wedge_degree 29]
                    [--n_neighbors 16]
                    [--fast_GF "False"]
                    [--require_global_correction "True"]
                    [--fusion_GF_kernel_size 49]
                    [--fusion_Gaussian_kernel_size 49]
                    [--X2_path None]
                    [--mask_path None]
                    [--boundary None]
                    [--save_path_top_or_left_view None]
                    [--save_path_bottom_or_right_view None]
```
## Installation

**Stable Release:** `pip install lsfm_destripe`<br>
**Development Head:** `pip install git+https://github.com/peng-lab/lsfm_destripe.git`

## Documentation

For full package documentation please visit [peng-lab.github.io/lsfm_destripe](https://peng-lab.github.io/lsfm_destripe).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.



**MIT license**

