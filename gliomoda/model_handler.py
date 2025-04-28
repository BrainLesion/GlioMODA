from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

import nibabel as nib
import numpy as np
import torch
from loguru import logger
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from gliomoda.constants import InferenceMode
from gliomoda.weights import check_weights_path


class ModelHandler:
    """Class for model loading, inference and post processing"""

    def __init__(self, device: torch.device) -> "ModelHandler":
        """Initialize the ModelHandler class.

        Args:
            device (torch.device): Device to use for inference.

        Returns:
            ModelHandler: ModelHandler instance.
        """

        self.device = device
        # Will be set during infer() call
        self.predictor = None
        self.inference_mode = None
        self.use_ResEncL = False

        # get location of model weights
        self.model_weights_folder = check_weights_path()

    def load_model(self, inference_mode: InferenceMode, use_ResEncL: bool) -> None:
        """Load the model for inference based on the inference mode

        Args:
            inference_mode (InferenceMode): inference mode (determined by passed images)
            use_ResEncL (bool): Whether to use ResEncL model or not. Only available if all 4 images are provided.
        """

        if (
            not self.predictor
            or self.inference_mode != inference_mode
            or self.use_ResEncL != use_ResEncL
        ):
            logger.debug(
                f"No loaded compatible model found (Switching from {self.inference_mode},{self.use_ResEncL=} to {inference_mode},{use_ResEncL=}). Loading Model and weights..."
            )
            if use_ResEncL and inference_mode != InferenceMode.T1C_T2F_T1N_T2W:
                logger.warning(
                    "ResEncL model is only available when providing all 4 modalities. Using default model instead."
                )
                use_ResEncL = False

            self.inference_mode = inference_mode
            self.use_ResEncL = use_ResEncL

            self.predictor = nnUNetPredictor(
                device=torch.device(self.device),
            )
            model_name = (
                self.model_weights_folder
                / f"{self.inference_mode.value}{'_ResEncL' if use_ResEncL else''}"
            )
            self.predictor.initialize_from_trained_model_folder(
                model_name,
                use_folds=("all"),
            )

            logger.debug(f"Successfully loaded model.")
        else:
            logger.debug(
                f"Same inference mode ({self.inference_mode}) as previous infer call. Re-using loaded model"
            )

    def infer(
        self,
        input_file_paths: List[Path],
        segmentation_file: Optional[str | Path] = None,
    ) -> np.ndarray:
        """Run inference on the provided images and save the segmentations to disk if paths are provided.

        Args:
            input_file_paths (List[Path]): _description_
            segmentation_file (Optional[str | Path], optional): Path to save segmentation file. Defaults to None.

        Returns:
            np.ndarray: Inferred segmentation.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            str_paths = [str(f) for f in input_file_paths]
            self.predictor.predict_from_files(
                [str_paths],
                tmpdir,
            )

            nifti_file = next(Path(tmpdir).glob("*.nii.gz"))
            segmentation_np = nib.load(nifti_file).get_fdata()

            # move segmentation to specified path
            if segmentation_file is not None:
                path = Path(segmentation_file)
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(nifti_file, path)
                logger.debug(f"Saved segmentation to {path}")

            return segmentation_np
