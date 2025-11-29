# GlioMODA

[![Python Versions](https://img.shields.io/pypi/pyversions/GlioMODA)](https://pypi.org/project/GlioMODA/)
[![Stable Version](https://img.shields.io/pypi/v/GlioMODA?label=stable)](https://pypi.python.org/pypi/GlioMODA/)
[![Documentation Status](https://readthedocs.org/projects/GlioMODA/badge/?version=latest)](http://GlioMODA.readthedocs.io/?badge=latest)
[![tests](https://github.com/BrainLesion/GlioMODA/actions/workflows/tests.yml/badge.svg)](https://github.com/BrainLesion/GlioMODA/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/BrainLesion/GlioMODA/graph/badge.svg?token=A7FWUKO9Y4)](https://codecov.io/gh/BrainLesion/GlioMODA)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<!-- ## Features -->

## Installation

With a Python 3.10+ environment, you can install `gliomoda` directly from [PyPI](https://pypi.org/project/gliomoda/):

```bash
pip install gliomoda
```


## Data Requirements

GlioMODA is trained on [BraTS](https://github.com/BraTS) (Brain Tumor Segmentation) preprocessed images.
This preprocessing typically includes co-registration to the T1c, skull stripping (brain extraction), and registration to the SRI-24 brain atlas (template).

We recommend using the [preprocessing package](https://github.com/BrainLesion/preprocessing), part of the [BrainLesion Suite](https://github.com/BrainLesion), to design custom preprocessing pipelines tailored to your specific needs.

Alternatively, if you have the full set of MRI modalities (T1, T1c, T2, FLAIR), you can use glioma-specific preprocessing functions from the [BraTS Orchestrator](https://github.com/BrainLesion/BraTS).


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
> [!IMPORTANT]
> This package is part of the [BrainLesion Suite](https://github.com/BrainLesion).   
> If you use GlioMODA in your research, please cite both the GlioMODA paper and the [BrainLesion Suite manuscript](https://github.com/BrainLesion#-citing-brainlesion-suite) to support the development.

Canisius, J., Buchner, J., Rosier, M., Griessmair, M., Peeken, J., Kirschke, J. S., Piraud, M., Bakas, S., Menze, B., Wiestler, B., & Kofler, F. (2025). GlioMODA: Robust Glioma Segmentation in Clinical Routine (p. 2025.11.12.25339968). medRxiv. https://doi.org/10.1101/2025.11.12.25339968

```
 @article{Canisius_Buchner_Rosier_Griessmair_Peeken_Kirschke_Piraud_Bakas_Menze_Wiestler_et al._2025, title={GlioMODA: Robust Glioma Segmentation in Clinical Routine}, rights={© 2025, Posted by Cold Spring Harbor Laboratory. This pre-print is available under a Creative Commons License (Attribution-NonCommercial 4.0 International), CC BY-NC 4.0, as described at http://creativecommons.org/licenses/by-nc/4.0/}, ISSN={3067-2007}, url={https://www.medrxiv.org/content/10.1101/2025.11.12.25339968v1}, DOI={10.1101/2025.11.12.25339968}, publisher={medRxiv}, author={Canisius, Julian and Buchner, Josef and Rosier, Marcel and Griessmair, Michael and Peeken, Jan and Kirschke, Jan S. and Piraud, Marie and Bakas, Spyridon and Menze, Bjoern and Wiestler, Benedikt and Kofler, Florian}, year={2025}, month=nov, pages={2025.11.12.25339968}, language={en} }
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
