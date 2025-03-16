from pathlib import Path
from unittest.mock import MagicMock, call, patch

import nibabel as nib
import numpy as np
import pytest

from gliomoda.constants import ATLAS_SPACE_SHAPE, DataMode, InferenceMode
from gliomoda.data_handler import DataHandler


@pytest.fixture
def data_handler():
    return DataHandler()


def test_validate_images_numpy(data_handler):
    """Test validation of NumPy input images."""
    img = np.zeros(ATLAS_SPACE_SHAPE, dtype=np.float32)

    images = data_handler.validate_images(t1c=img, t2f=img)
    assert data_handler.input_mode == DataMode.NUMPY
    assert all(isinstance(i, np.ndarray) for i in images if i is not None)


@patch("gliomoda.data_handler.nib.load")
def test_validate_images_nifti(mock_nib_load, data_handler, tmp_path):
    """Test validation of NIfTI input images."""
    mock_img = MagicMock()
    mock_img.get_fdata.return_value = np.zeros(ATLAS_SPACE_SHAPE, dtype=np.float32)
    mock_nib_load.return_value = mock_img

    nifti_file = tmp_path / "test.nii.gz"
    nifti_file.touch()  # Create an empty file

    images = data_handler.validate_images(t1c=nifti_file, t2f=nifti_file)
    assert data_handler.input_mode == DataMode.NIFTI_FILE
    assert all(isinstance(i, Path) for i in images if i is not None)


def test_validate_images_shape_mismatch(data_handler):
    """Test validation failure due to shape mismatch."""
    atlas_img = np.zeros(ATLAS_SPACE_SHAPE, dtype=np.float32)
    non_atlas_img = np.zeros((200, 200, 200), dtype=np.float32)
    with pytest.raises(AssertionError, match="Invalid shape for input data"):
        data_handler.validate_images(t1c=atlas_img, t2w=non_atlas_img)


@patch("gliomoda.data_handler.nib.load")
def test_validate_images_type_mismatch(mock_nib_load, data_handler, tmp_path):
    """Test failure when mixing NumPy and NIfTI inputs."""
    img = np.zeros(ATLAS_SPACE_SHAPE, dtype=np.float32)

    mock_img = MagicMock()
    mock_img.get_fdata.return_value = np.zeros(ATLAS_SPACE_SHAPE, dtype=np.float32)
    mock_nib_load.return_value = mock_img
    nifti_file = tmp_path / "file.nii.gz"
    nifti_file.touch()  # Create an empty file

    with pytest.raises(
        AssertionError, match="All passed images must be of the same type"
    ):
        data_handler.validate_images(t1c=img, t2f=nifti_file)


def test_validate_images_file_not_found(data_handler):
    """Test failure when mixing NumPy and NIfTI inputs."""
    with pytest.raises(FileNotFoundError, match="not found"):
        data_handler.validate_images(t1c="fake.nii.gz")


def test_validate_images_file_not_nifti(data_handler, tmp_path):

    file = tmp_path / "file.txt"
    file.touch()
    """Test failure when mixing NumPy and NIfTI inputs."""
    with pytest.raises(ValueError, match="must be a NIfTI file"):
        data_handler.validate_images(t1c=file)


def test_validate_images_no_files(data_handler):
    """Test failure when no files are passed."""
    with pytest.raises(AssertionError, match="No input images provided"):
        data_handler.validate_images()


def test_determine_inference_mode_not_validated_yet(data_handler):
    with pytest.raises(AssertionError, match="Please validate the input images first"):
        data_handler.determine_inference_mode([None, None, None, None])


def test_determine_inference_mode_invalid_combination(data_handler):
    data_handler.input_mode = DataMode.NUMPY
    with pytest.raises(
        NotImplementedError, match="No model implemented for this combination of images"
    ):
        data_handler.determine_inference_mode([None, None, None, None])


def test_determine_inference_mode_okay(data_handler):
    data_handler.input_mode = DataMode.NUMPY
    mode = data_handler.determine_inference_mode([Path("test"), None, None, None])
    assert mode == InferenceMode.T1C


@patch("gliomoda.data_handler.nib.save")
def test_get_input_file_paths_numpy(mock_nib_save, data_handler, tmp_path):
    """Test conversion of NumPy inputs to temporary NIfTI files."""
    img = np.zeros(ATLAS_SPACE_SHAPE, dtype=np.float32)

    data_handler.input_mode = DataMode.NUMPY
    files = data_handler.get_input_file_paths([img, None, img], tmp_folder=tmp_path)

    assert len(files) == 2
    assert all(isinstance(f, Path) for f in files)
    assert all(f.suffixes == [".nii", ".gz"] for f in files)
    assert mock_nib_save.call_count == 2


def test_get_input_file_paths_numpy_no_tmp_folder(data_handler):
    """Test conversion of NumPy inputs to temporary NIfTI files."""
    img = np.zeros(ATLAS_SPACE_SHAPE, dtype=np.float32)

    data_handler.input_mode = DataMode.NUMPY
    with pytest.raises(ValueError, match="Please provide a temporary folder"):
        data_handler.get_input_file_paths([img, None, img], tmp_folder=None)


def test_get_input_file_paths_numpy_invalid_mode(data_handler):
    """Test conversion of NumPy inputs to temporary NIfTI files."""
    img = np.zeros(ATLAS_SPACE_SHAPE, dtype=np.float32)

    data_handler.input_mode = "INVALID"
    with pytest.raises(NotImplementedError, match="Input mode"):
        data_handler.get_input_file_paths([img, None, img], tmp_folder=None)


def test_get_input_file_paths_nifti(data_handler, tmp_path):
    """Test returning file paths when input mode is NIfTI."""
    nifti_file = tmp_path / "test.nii.gz"
    nifti_file.touch()

    data_handler.input_mode = DataMode.NIFTI_FILE
    files = data_handler.get_input_file_paths([nifti_file, None])

    assert files == [nifti_file]
