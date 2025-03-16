from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import nibabel as nib
import numpy as np
from loguru import logger

from gliomoda.constants import (
    ATLAS_SPACE_SHAPE,
    IMGS_TO_MODE_DICT,
    DataMode,
    InferenceMode,
)


class DataHandler:
    """Class to perform data related tasks such as validation, loading, transformation, saving."""

    def __init__(self) -> "DataHandler":
        # Following will be inferred from the input images during validation
        self.input_mode = None
        self.reference_nifti_file = None

    @property
    def get_input_mode(self) -> DataMode:
        """Get the input mode.

        Returns:
            DataMode: Input mode.
        Raises:
            AssertionError: If the input mode is not set (i.e. input images were not validated)
        """
        assert (
            self.input_mode is not None
        ), "Input mode not set. Please validate the input images first by calling .validate_images(...)."
        return self.input_mode

    @property
    def get_reference_nifti_file(self) -> Path | str:
        """Get a reference NIfTI file from the input (first not None in order T1-T1C-T2-FLAIR) to match header and affine.

        Returns:
            Path: Path to reference NIfTI file.
        Raises:
            AssertionError: If the reference NIfTI file is not set (i.e. input images were not validated)
        """
        assert (
            self.reference_nifti_file is not None
        ), "Reference NIfTI file not set. Please ensure you provided paths to NIfTI images and validated the input images first by calling .validate_images(...)."
        return self.reference_nifti_file

    def _validate_shape(self, data: np.ndarray) -> None:
        """Validate the shape of the input data.

        Args:
            data (np.ndarray): Input data.
        Raises:
            AssertionError: If the shape of the input data does not match the standard atlas space shape.
        """
        assert (
            data.shape == ATLAS_SPACE_SHAPE
        ), f"Invalid shape for input data, should match standard atlas space shape: {ATLAS_SPACE_SHAPE}"

    def validate_images(
        self,
        t1c: Optional[str | Path | np.ndarray] = None,
        t2f: Optional[str | Path | np.ndarray] = None,
        t1n: Optional[str | Path | np.ndarray] = None,
        t2w: Optional[str | Path | np.ndarray] = None,
    ) -> List[np.ndarray | None] | List[Path | None]:
        """Validate inputs. \n
        Verify that the input images
            - exist (for paths)
            - are all of the same type (NumPy or NIfTI).
            - have the correct shape (ATLAS_SPACE_SHAPE).

        Sets internal variables input_mode and reference_nifti_file.

        Args:
            t1c (Optional[str | Path | np.ndarray], optional): T1C image. Defaults to None.
            t2f (Optional[str | Path | np.ndarray], optional): T2F image. Defaults to None.
            t1n (Optional[str | Path | np.ndarray], optional): T1N image. Defaults to None.
            t2w (Optional[str | Path | np.ndarray], optional): T2W image. Defaults to None.

        Returns:
            List[np.ndarray | None] | List[Path | None]: List of validated images.

        Raises:
            FileNotFoundError: If a file is not found.
            ValueError: If a file path is not a NIfTI file (.nii or .nii.gz).
            AssertionError: If no input images are provided.
            AssertionError: If the provided images are not of the same type.
        """

        def _validate_image(
            data: str | Path | np.ndarray | None,
        ) -> np.ndarray | Path | None:
            if data is None:
                return None
            if isinstance(data, np.ndarray):
                self.input_mode = DataMode.NUMPY
                self._validate_shape(data)
                return data.astype(np.float32)

            data = Path(data)
            if not data.exists():
                raise FileNotFoundError(f"File {data} not found")

            if not (data.suffix == ".nii" or data.suffixes == [".nii", ".gz"]):
                raise ValueError(
                    f"File {data} must be a NIfTI file with extension .nii or .nii.gz"
                )

            self._validate_shape(nib.load(data).get_fdata())

            self.input_mode = DataMode.NIFTI_FILE
            return data.absolute()

        images = [
            _validate_image(img)
            for img in [
                t1c,
                t2f,
                t1n,
                t2w,
            ]
        ]
        not_none_images = [img for img in images if img is not None]
        assert len(not_none_images) > 0, "No input images provided"
        # make sure all inputs have the same type
        unique_types = set(map(type, not_none_images))
        assert (
            len(unique_types) == 1
        ), f"All passed images must be of the same type! Received {unique_types}. Accepted Input types: {list(DataMode)}"

        if self.input_mode is DataMode.NIFTI_FILE:
            self.reference_nifti_file = not_none_images[0]
        logger.info(
            f"Successfully validated input images (received {len(not_none_images)}). Input mode: {self.input_mode}"
        )
        return images

    def determine_inference_mode(
        self, images: List[np.ndarray | None] | List[Path | None]
    ) -> InferenceMode:
        """Determine the inference mode based on the provided images.
        Args:
            images (List[np.ndarray | None] | List[Path | None]): List of validated images.
        Returns:
            InferenceMode: Inference mode based on the combination of input images.
        Raises:
            NotImplementedError: If no model is implemented for the combination of input images.
            AssertionError: If the input mode is not set (i.e. input images were not validated)
        """
        assert (
            self.input_mode is not None
        ), "Please validate the input images first by calling validate_images(...)."

        _t1c, _flair, _t1, _t2 = [img is not None for img in images]
        logger.debug(
            f"Received files: T1C: {_t1c}, FLAIR: {_flair}, T1: {_t1}, T2: {_t2}"
        )
        # check if files are given in a valid combination that has an existing model implementation
        mode = IMGS_TO_MODE_DICT.get((_t1c, _flair, _t1, _t2), None)
        if mode is None:
            raise NotImplementedError(
                f"No model implemented for this combination of images: T1C: {_t1c}, FLAIR: {_flair}, T1: {_t1}, T2: {_t2}. {os.linesep}Available models: {[mode.value for mode in InferenceMode]}"
            )
        logger.info(f"Inference mode: {mode}")
        return mode

    def get_input_file_paths(
        self,
        images: List[np.ndarray | None] | List[Path | None],
        tmp_folder: Optional[Path] = None,
    ) -> List[np.ndarray]:
        """Load the input images based on the input mode.

        Args:
            images (List[np.ndarray | None] | List[Path | None]): List of validated images.
        Returns:
            List[Path]: List of input images (NIFTI paths).
        Raises:
            AssertionError: If the input mode is not set (i.e. input images were not validated)
            ValueError: If the input mode is NumPy and no temporary folder is provided.
            NotImplementedError: If the input mode is not implemented.
        """
        assert (
            self.input_mode is not None
        ), "Please validate the input images first by calling validate_images(...)."

        if self.input_mode == DataMode.NIFTI_FILE:
            return [img for img in images if img is not None]
        elif self.input_mode == DataMode.NUMPY:
            if tmp_folder is None:
                raise ValueError(
                    "Please provide a temporary folder when using NumPy input mode."
                )
            input_files = []
            for idx, img in enumerate(images):
                if img is not None:
                    input_file = tmp_folder / f"input_{idx}.nii.gz"
                    nib.save(nib.Nifti1Image(img, np.eye(4)), input_file)
                    input_files.append(input_file)
            return input_files
        else:
            raise NotImplementedError(
                f"Input mode {self.input_mode} not implemented. Available modes: {list(DataMode)}"
            )
