from enum import Enum
from pathlib import Path


class InferenceMode(str, Enum):
    """
    Enum representing different modes of inference based on available image inputs.
    In general, you should aim to use as many modalities as possible to get the best results.
    """

    T1C_T2F_T1N_T2W = "t1c_t2f_t1n_t2w"
    """T1C, T2F, T1N, and T2W are available."""

    T1C = "t1c"
    """T1C is available."""

    T2F = "t2f"
    """T2F is available."""

    T1N = "t1n"
    """T1N is available."""

    T2W = "t2w"
    """T2W is available."""

    T1C_T2F = "t1c_t2f"
    """T1C and T2F are available."""

    T1C_T1N = "t1c_t1n"
    """T1C and T1N are available."""

    T1C_T1N_T2W = "t1c_t1n_t2w"
    """T1C, T1N, and T2W are available."""

    T1C_T2W = "t1c_t2w"
    """T1C and T2W are available."""

    T2F_T1N = "t2f_t1n"
    """T2F and T1N are available."""

    T2F_T1N_T2W = "t2f_t1n_t2w"
    """T2F, T1N, and T2W are available."""

    T2F_T2W = "t2f_t2w"
    """T2F and T2W are available."""


class DataMode(str, Enum):
    """Enum representing different modes for handling input and output data."""

    NIFTI_FILE = "NIFTI_FILEPATH"
    """Input data is provided as NIFTI file paths/ output is written to NIFTI files."""
    NUMPY = "NP_NDARRAY"
    """Input data is provided as NumPy arrays/ output is returned as NumPy arrays."""


MODALITIES = ["t1c", "fla", "t1", "t2"]
"""List of modality names in standard order: T1C, FLAIR, T1, T2."""


# booleans indicate presence of files in order: T1C, FLAIR, T1, T2
IMGS_TO_MODE_DICT = {
    (True, True, True, True): InferenceMode.T1C_T2F_T1N_T2W,
    (True, False, False, False): InferenceMode.T1C,
    (False, True, False, False): InferenceMode.T2F,
    (False, False, True, False): InferenceMode.T1N,
    (False, False, False, True): InferenceMode.T2W,
    (True, True, False, False): InferenceMode.T1C_T2F,
    (True, False, True, False): InferenceMode.T1C_T1N,
    (True, False, True, True): InferenceMode.T1C_T1N_T2W,
    (True, False, False, True): InferenceMode.T1C_T2W,
    (False, True, True, False): InferenceMode.T2F_T1N,
    (False, True, True, True): InferenceMode.T2F_T1N_T2W,
    (False, True, False, True): InferenceMode.T2F_T2W,
}

"""Dictionary mapping tuples of booleans representing presence of the modality in order [T1C, FLAIR, T1, T2] to InferenceMode values."""

ZENODO_RECORD_URL = "https://zenodo.org/api/records/14989866"
WEIGHTS_FOLDER = Path(__file__).parent / "weights"
WEIGHTS_DIR_PATTERN = "weights_v*.*.*"
"""Directory name pattern to store model weights. E.g. weights_v1.0.0"""

ATLAS_SPACE_SHAPE = (240, 240, 155)
"""Standard shape of the atlas space."""

NNUNET_ENV_VARS = [
    "nnUNet_raw",
    "nnUNet_preprocessed",
    "nnUNet_results",
]
