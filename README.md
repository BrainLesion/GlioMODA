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


## Data Requirements

GlioMODA is trained on [BraTS](https://github.com/BraTS) (Brain Tumor Segmentation) preprocessed images.
This preprocessing typically includes co-registration to the T1C, skull stripping (brain extraction), and registration to the SRI-24 brain atlas (template).

We recommend using the [preprocessing package](https://github.com/BrainLesion/preprocessing), part of the [BrainLesion Suite](https://github.com/BrainLesion), to design custom preprocessing pipelines tailored to your specific needs.
You can install the package with:

```bash
pip install brainles-preprocessing
```

Alternatively, if you have the full set of MRI modalities (T1, T1c, T2, T2-FLAIR), you can use glioma-specific preprocessing functions from the [BraTS Orchestrator](https://github.com/BrainLesion/BraTS).
Install with:

```bash
pip install brats[preprocessing]
```


## Use Cases and Tutorials

A minimal example to create a segmentation could look like this:

```python
from gliomoda import Inferer

inferer = Inferer()

# Save NIfTI files
inferer.infer(
    t1c="path/to/t1c.nii.gz",
    t2f="path/to/t2f.nii.gz",
    t1n="path/to/t1n.nii.gz",
    t2w="path/to/t2w.nii.gz",
    segmentation_file="path/to/segmentation.nii.gz",
)

# Or directly use pre-loaded NumPy data. (Both works as well)
segmentation_np = inferer.infer(
    t1c=t1c_np,
    t2f=t2f_np,
    t1n=t1n_np,
    t2w=t2w_np,
)
```
> [!NOTE] 
>If you're interested in the GlioMODA package, the [BraTS Adult Glioma Segmentation](https://github.com/BrainLesion/BraTS?tab=readme-ov-file#adult-glioma-segmentation-pre-treatment) may also be of interest.

<!-- For more examples and details please refer to our extensive Notebook tutorials here [NBViewer](https://nbviewer.org/github/BrainLesion/tutorials/blob/main/GlioMODA/tutorial.ipynb) ([GitHub](https://github.com/BrainLesion/tutorials/blob/main/GlioMODA/tutorial.ipynb)). For the best experience open the notebook in Colab. -->


## Citation

If you use GlioMODA in your research, please cite it to support the development!

```
TODO: citation will be added asap
```

## Trouble shoot

<details>
<summary>
Multiprocessing error
</summary>

If you get an error related to something like this:
<br>

```
RuntimeError: 
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.

        To fix this issue, refer to the "Safe importing of main module"
        section in https://docs.python.org/3/library/multiprocessing.html
```

Please ensure you properly wrap your script:

```python
if __name__ == "__main__":
    inferer = Inferer()
    ...
```

</details>



## Contributing

We welcome all kinds of contributions from the community!

### Reporting Bugs, Feature Requests and Questions

Please open a new issue [here](https://github.com/BrainLesion/GlioMODA/issues).

### Code contributions

Nice to have you on board! Please have a look at our [CONTRIBUTING.md](CONTRIBUTING.md) file.
