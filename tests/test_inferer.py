from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gliomoda.inferer import Inferer


@pytest.fixture
@patch("gliomoda.inferer.DataHandler")
@patch("gliomoda.inferer.ModelHandler")
def inferer(mock_model_handler, mock_data_handler):
    return Inferer(device="cpu")


@patch("gliomoda.inferer.torch.cuda.is_available", return_value=True)
@patch("gliomoda.inferer.DataHandler")
@patch("gliomoda.inferer.ModelHandler")
def test_device_configuration_gpu(
    mock_model_handler, mock_data_handler, mock_cuda_is_available
):
    inferer_gpu = Inferer(device="cuda", cuda_visible_devices="0")
    assert inferer_gpu.device.type == "cuda"


@patch("gliomoda.inferer.DataHandler")
@patch("gliomoda.inferer.ModelHandler")
def test_device_configuration_cpu(mock_model_handler, mock_data_handler):
    inferer_cpu = Inferer(device="cpu")
    assert inferer_cpu.device.type == "cpu"


def test_infer(inferer):
    inferer.infer(t1c="test")
