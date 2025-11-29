from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from loguru import logger

from gliomoda.data_handler import DataHandler
from gliomoda.model_handler import ModelHandler
from gliomoda.utils import citation_reminder


class Inferer:

    def __init__(
        self,
        device: Optional[str] = "cuda",
        cuda_visible_devices: Optional[str] = "0",
    ) -> None:
        """
        Initialize the Inferer class.

        Args:
            device (Optional[str], optional): torch device string. Defaults to "cuda".
            cuda_visible_devices (Optional[str], optional): CUDA_VISIBLE_DEVICES environment variable. Defaults to "0".
        """
        self.device = self._configure_device(
            requested_device=device,
            cuda_visible_devices=cuda_visible_devices,
        )
        self.data_handler = DataHandler()
        self.model_handler = ModelHandler(device=self.device)

    def _configure_device(
        self, requested_device: str, cuda_visible_devices: str
    ) -> torch.device:
        """Configure the device for inference based on the specified config.device.

        Args:
            requested_device (str): Requested device.
            cuda_visible_devices (str): CUDA_VISIBLE_DEVICES environment variable.

        Returns:
            torch.device: Configured device.
        """
        device = torch.device(requested_device)
        if device.type == "cuda":
            # The env vars have to be set before the first call to torch.cuda, else torch will always attempt to use the first device
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
            if torch.cuda.is_available():
                # clean memory
                torch.cuda.empty_cache()

        logger.info(f"Set torch device: {device}")

        return device

    @citation_reminder
    def infer(
        self,
        t1c: Optional[str | Path | np.ndarray] = None,
        t2f: Optional[str | Path | np.ndarray] = None,
        t1n: Optional[str | Path | np.ndarray] = None,
        t2w: Optional[str | Path | np.ndarray] = None,
        segmentation_file: Optional[str | Path] = None,
        use_ResEncL: bool = False,
    ) -> np.ndarray:
        """Infer segmentations based on provided images.

        Args:
            t1c (Optional[str  |  Path  |  np.ndarray], optional): T1C image. Defaults to None.
            t2f (Optional[str  |  Path  |  np.ndarray], optional): T2F image. Defaults to None.
            t1n (Optional[str  |  Path  |  np.ndarray], optional): T1N image. Defaults to None.
            t2w (Optional[str  |  Path  |  np.ndarray], optional): T2W image. Defaults to None.
            segmentation_file (Optional[str  |  Path], optional): Segmentation file. Defaults to None.
            use_ResEncL (bool, optional): Use ResEncL model (only available when providing all 4 modalities). Defaults to False.

        Returns:
            np.ndarray: Inferred segmentation.
        """

        # check inputs and get mode , if mode == prev mode => run inference, else load new model
        validated_images = self.data_handler.validate_images(
            t1c=t1c,
            t2f=t2f,
            t1n=t1n,
            t2w=t2w,
        )
        determined_inference_mode = self.data_handler.determine_inference_mode(
            images=validated_images
        )

        self.model_handler.load_model(
            inference_mode=determined_inference_mode,
            use_ResEncL=use_ResEncL,
        )

        with tempfile.TemporaryDirectory() as tmpdir:

            input_file_paths = self.data_handler.get_input_file_paths(
                images=validated_images,
                tmp_folder=Path(tmpdir),
            )

            logger.info(f"Running inference on device: {self.device}")
            np_results = self.model_handler.infer(
                input_file_paths=input_file_paths,
                segmentation_file=segmentation_file,
            )
            logger.info(f"Finished inference")
            return np_results
