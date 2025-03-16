# GlioMODA

[![Python Versions](https://img.shields.io/pypi/pyversions/GlioMODA)](https://pypi.org/project/GlioMODA/)
[![Stable Version](https://img.shields.io/pypi/v/GlioMODA?label=stable)](https://pypi.python.org/pypi/GlioMODA/)
[![Documentation Status](https://readthedocs.org/projects/GlioMODA/badge/?version=latest)](http://GlioMODA.readthedocs.io/?badge=latest)
[![tests](https://github.com/BrainLesion/GlioMODA/actions/workflows/tests.yml/badge.svg)](https://github.com/BrainLesion/GlioMODA/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/BrainLesion/GlioMODA/graph/badge.svg?token=A7FWUKO9Y4)](https://codecov.io/gh/BrainLesion/GlioMODA)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Features


## Installation

With a Python 3.10+ environment, you can install `gliomoda` directly from [PyPI](https://pypi.org/project/gliomoda/):

```bash
pip install gliomoda
```


## Use Cases and Tutorials

A minimal example to create a segmentation could look like this:

```python
from gliomoda import Inferer

inferer = Inferer()

# Save NIfTI files
inferer.infer(
    t1c="path/to/t1c.nii.gz",
    fla="path/to/fla.nii.gz",
    t1="path/to/t1.nii.gz",
    t2="path/to/t2.nii.gz",
    segmentation_file="example/seg.nii.gz",
)

# Or directly use NumPy data. (Both works as well)
segmentation_np = inferer.infer(
    t1c="path/to/t1c.nii.gz",
    fla="path/to/fla.nii.gz",
    t1="path/to/t1.nii.gz",
    t2="path/to/t2.nii.gz",◊
)
```

<!-- For more examples and details please refer to our extensive Notebook tutorials here [NBViewer](https://nbviewer.org/github/BrainLesion/tutorials/blob/main/GlioMODA/tutorial.ipynb) ([GitHub](https://github.com/BrainLesion/tutorials/blob/main/GlioMODA/tutorial.ipynb)). For the best experience open the notebook in Colab. -->


## Citation

If you use GlioMODA in your research, please cite it to support the development!

```
TODO: citation will be added asap
```

## Contributing

We welcome all kinds of contributions from the community!

### Reporting Bugs, Feature Requests and Questions

Please open a new issue [here](https://github.com/BrainLesion/GlioMODA/issues).

### Code contributions

Nice to have you on board! Please have a look at our [CONTRIBUTING.md](CONTRIBUTING.md) file.
