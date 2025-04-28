from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import torch

from gliomoda.constants import InferenceMode
from gliomoda.model_handler import ModelHandler


@pytest.fixture
@patch(
    "gliomoda.model_handler.check_weights_path",
    return_value=Path("/mocked/path/to/weights"),
)
def mock_model_handler(_):
    """Fixture to create a ModelHandler instance with a mocked device."""
    device = torch.device("cpu")
    handler = ModelHandler(device)
    return handler


@patch("gliomoda.model_handler.nnUNetPredictor")
def test_load_model(mock_predictor_class, mock_model_handler):
    """Test if load_model correctly initializes the nnUNetPredictor."""
    mock_predictor = MagicMock()
    mock_predictor_class.return_value = mock_predictor

    inference_mode = InferenceMode.T1N
    mock_model_handler.load_model(inference_mode, use_ResEncL=False)

    assert mock_model_handler.predictor is not None
    # mock_predictor.initialize_from_trained_model_folder
    mock_predictor.initialize_from_trained_model_folder.assert_called_once_with(
        Path("/mocked/path/to/weights") / inference_mode.value,
        use_folds=("all"),
    )


@patch("gliomoda.model_handler.logger.warning")
@patch("gliomoda.model_handler.nnUNetPredictor")
def test_load_model_useResEncL_invalid(
    mock_predictor_class, mock_logger_warning, mock_model_handler
):
    """Test if load_model correctly initializes the nnUNetPredictor."""
    mock_predictor = MagicMock()
    mock_predictor_class.return_value = mock_predictor

    inference_mode = InferenceMode.T1N
    mock_model_handler.load_model(inference_mode, use_ResEncL=True)

    assert mock_model_handler.predictor is not None
    # mock_predictor.initialize_from_trained_model_folder
    mock_predictor.initialize_from_trained_model_folder.assert_called_once_with(
        Path("/mocked/path/to/weights") / inference_mode.value,
        use_folds=("all"),
    )

    mock_logger_warning.assert_called_once_with(
        "ResEncL model is only available when providing all 4 modalities. Using default model instead."
    )


@patch("gliomoda.model_handler.logger.warning")
@patch("gliomoda.model_handler.nnUNetPredictor")
def test_load_model_useResEncL_valid(
    mock_predictor_class, mock_logger_warning, mock_model_handler
):
    """Test if load_model correctly initializes the nnUNetPredictor."""
    mock_predictor = MagicMock()
    mock_predictor_class.return_value = mock_predictor

    inference_mode = InferenceMode.T1C_T2F_T1N_T2W
    mock_model_handler.load_model(inference_mode, use_ResEncL=True)

    assert mock_model_handler.predictor is not None
    # mock_predictor.initialize_from_trained_model_folder
    mock_predictor.initialize_from_trained_model_folder.assert_called_once_with(
        Path("/mocked/path/to/weights") / f"{inference_mode.value}_ResEncL",
        use_folds=("all"),
    )

    mock_logger_warning.assert_not_called()


@patch("gliomoda.model_handler.nib.load")
@patch("gliomoda.model_handler.Path.glob", return_value=iter([Path("file1.nii.gz")]))
@patch("gliomoda.model_handler.nnUNetPredictor")
@patch("gliomoda.model_handler.shutil.move")
def test_infer_not_saving(
    mock_move,
    mock_predictor_class,
    mock_glob,
    mock_nib_load,
    mock_model_handler,
    tmp_path,
):
    """Test if infer runs inference and saves output correctly."""
    mock_predictor = MagicMock()
    mock_predictor_class.return_value = mock_predictor

    # Mock nifti loading
    mock_nifti_img = MagicMock()
    mock_nifti_img.get_fdata.return_value = np.random.rand(10, 10, 10)
    mock_nib_load.return_value = mock_nifti_img

    mock_model_handler.predictor = mock_predictor
    input_files = [tmp_path / "image1.nii.gz", tmp_path / "image2.nii.gz"]

    seg = mock_model_handler.infer(input_files)

    mock_move.assert_not_called()


@patch("gliomoda.model_handler.nib.load")
@patch("gliomoda.model_handler.Path.glob", return_value=iter([Path("file1.nii.gz")]))
@patch("gliomoda.model_handler.nnUNetPredictor")
@patch("gliomoda.model_handler.shutil.move")
def test_infer_saving(
    mock_move,
    mock_predictor_class,
    mock_glob,
    mock_nib_load,
    mock_model_handler,
    tmp_path,
):
    """Test if infer runs inference and returns output without saving a file."""
    mock_predictor = MagicMock()
    mock_predictor_class.return_value = mock_predictor

    # Mock nifti loading
    mock_nifti_img = MagicMock()
    mock_nifti_img.get_fdata.return_value = np.random.rand(10, 10, 10)
    mock_nib_load.return_value = mock_nifti_img

    mock_model_handler.predictor = mock_predictor
    input_files = [tmp_path / "image1.nii.gz", tmp_path / "image2.nii.gz"]

    seg_path = "segmentation.nii.gz"
    mock_model_handler.infer(input_files, seg_path)

    mock_move.assert_called_once()
