# LSFM DeStripe PyTorch

[![Build Status](https://github.com/peng-lab/lsfm_destripe/workflows/Build%20Main/badge.svg)](https://github.com/peng-lab/lsfm_destripe/actions)
[![Documentation](https://github.com/peng-lab/lsfm_destripe/workflows/Documentation/badge.svg)](https://peng-lab.github.io/lsfm_destripe/)
[![Code Coverage](https://codecov.io/gh/peng-lab/lsfm_destripe/branch/main/graph/badge.svg)](https://codecov.io/gh/peng-lab/lsfm_destripe)

A PyTorch implementation of LSFM DeStripe method

(add a short description, to be updated by Yu)

---

## Quick Start

### Use as Python API (TO be updated by Yu)
(1) Provide a numpy array, i.e., the image to be processed, and necessary parameters (more suitable for small data or for use in napari)
```python
from lsfm_destripe import DeStripe

out = DeStripe.train_full_arr(img_arr, mask_arr, is_vertical, train_param, device, qr, require_global_correction)
```
(2) Provide a filename, run slice by slice (suitable for extremely large file)
```python
from lsfm_destripe import DeStripe

exe = DeStripe()
# run with default parameters
exe = DeStripe(data_path)
# adjust some parameters
exe = DeStripe(data_path, isVertical, angleOffset,losseps, mask_name)
out = exe.train()
```

### Run from command line for batch processing (To be updated by Yu)
(1) run with all default parameters
```bash
destripe --data_path /path/to/my/image.tiff --save_path /path/to/save/results
```

(2) run with different parameters
```bash
destripe --data_path /path/to/my/image.tiff \
         --save_path /path/to/save/results \
         --deg 12 \
         --Nneighbors 32 \
         --n_epochs 500
```


## Installation

**Stable Release:** `pip install lsfm_destripe`<br>
**Development Head:** `pip install git+https://github.com/peng-lab/lsfm_destripe.git`

## Documentation

For full package documentation please visit [peng-lab.github.io/lsfm_destripe](https://peng-lab.github.io/lsfm_destripe).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.



**MIT license**

