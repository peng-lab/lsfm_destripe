# LSFM DeStripe PyTorch

[![Build Status](https://github.com/peng-lab/lsfm_destripe/workflows/Build%20Main/badge.svg)](https://github.com/peng-lab/lsfm_destripe/actions)
[![Documentation](https://github.com/peng-lab/lsfm_destripe/workflows/Documentation/badge.svg)](https://peng-lab.github.io/lsfm_destripe/)
[![Code Coverage](https://codecov.io/gh/peng-lab/lsfm_destripe/branch/main/graph/badge.svg)](https://codecov.io/gh/peng-lab/lsfm_destripe)

A PyTorch implementation of LSFM DeStripe method

---

## Quick Start

### Use as Python API
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

### Run from command line for batch processing
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

#### Additional Optional Setup Steps:

-   Turn your project into a GitHub repository:
    -   Make an account on [github.com](https://github.com)
    -   Go to [make a new repository](https://github.com/new)
    -   _Recommendations:_
        -   _It is strongly recommended to make the repository name the same as the Python
            package name_
        -   _A lot of the following optional steps are *free* if the repository is Public,
            plus open source is cool_
    -   After a GitHub repo has been created, run the commands listed under:
        "...or push an existing repository from the command line"
-   Register your project with Codecov:
    -   Make an account on [codecov.io](https://codecov.io)(Recommended to sign in with GitHub)
        everything else will be handled for you.
-   Ensure that you have set GitHub pages to build the `gh-pages` branch by selecting the
    `gh-pages` branch in the dropdown in the "GitHub Pages" section of the repository settings.
    ([Repo Settings](https://github.com/peng-lab/lsfm_destripe/settings))
-   Register your project with PyPI:
    -   Make an account on [pypi.org](https://pypi.org)
    -   Go to your GitHub repository's settings and under the
        [Secrets tab](https://github.com/peng-lab/lsfm_destripe/settings/secrets/actions),
        add a secret called `PYPI_TOKEN` with your password for your PyPI account.
        Don't worry, no one will see this password because it will be encrypted.
    -   Next time you push to the branch `main` after using `bump2version`, GitHub
        actions will build and deploy your Python package to PyPI.




**MIT license**

