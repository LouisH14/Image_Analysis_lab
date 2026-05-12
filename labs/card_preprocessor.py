"""
Reusable preprocessing module for UNO card images.

Provides functions to apply HSV thresholding, morphological closing,
and small component removal to improve robustness to surface defects.
"""

from typing import Optional, Tuple

import cv2
import numpy as np


class CardPreprocessor:
    """Configure and apply HSV-based card preprocessing."""

    def __init__(
        self,
        lower_hsv: Tuple[int, int, int] = (0, 0, 0),
        upper_hsv: Tuple[int, int, int] = (179, 255, 255),
        closing_kernel_size: int = 5,
        min_component_area: int = 250,
        enabled: bool = False,
    ):
        """
        Initialize preprocessor.

        Parameters
        ----------
        lower_hsv : Tuple[int, int, int]
            Lower HSV bound (H: 0-179, S/V: 0-255). Default accepts all colors.
        upper_hsv : Tuple[int, int, int]
            Upper HSV bound. Default accepts all colors.
        closing_kernel_size : int
            Kernel size for morphological closing (must be odd, ≥ 1).
        min_component_area : int
            Minimum pixel area to keep a connected component. Default 250.
        enabled : bool
            Whether to enable preprocessing by default.
        """
        self.lower_hsv = np.array(lower_hsv, dtype=np.uint8)
        self.upper_hsv = np.array(upper_hsv, dtype=np.uint8)
        self.closing_kernel_size = max(1, closing_kernel_size)
        if self.closing_kernel_size % 2 == 0:
            self.closing_kernel_size += 1
        self.min_component_area = max(0, min_component_area)
        self.enabled = enabled

    def preprocess(
        self, image_bgr: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply preprocessing to an image.

        Parameters
        ----------
        image_bgr : np.ndarray
            Input image in BGR format.

        Returns
        -------
        output_image : np.ndarray
            Preprocessed image (same format as input if not enabled, else masked).
        mask : np.ndarray or None
            Binary mask used for preprocessing (for visualization). None if not enabled.
        """
        if not self.enabled:
            return image_bgr, None

        # Convert to HSV
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        # Apply HSV threshold
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)

        # Apply morphological closing
        if self.closing_kernel_size > 1:
            kernel = np.ones(
                (self.closing_kernel_size, self.closing_kernel_size), dtype=np.uint8
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Remove small components
        if self.min_component_area > 0:
            mask = self._remove_small_components(mask, self.min_component_area)

        # Apply mask to image
        output_image = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)

        return output_image, mask

    @staticmethod
    def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
        """Remove connected components smaller than min_area."""
        if min_area <= 0:
            return mask

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        cleaned = np.zeros_like(mask)

        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= min_area:
                cleaned[labels == label] = 255

        return cleaned

    def set_parameters(
        self,
        lower_hsv: Optional[Tuple[int, int, int]] = None,
        upper_hsv: Optional[Tuple[int, int, int]] = None,
        closing_kernel_size: Optional[int] = None,
        min_component_area: Optional[int] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        """Update preprocessor parameters."""
        if lower_hsv is not None:
            self.lower_hsv = np.array(lower_hsv, dtype=np.uint8)
        if upper_hsv is not None:
            self.upper_hsv = np.array(upper_hsv, dtype=np.uint8)
        if closing_kernel_size is not None:
            self.closing_kernel_size = max(1, closing_kernel_size)
            if self.closing_kernel_size % 2 == 0:
                self.closing_kernel_size += 1
        if min_component_area is not None:
            self.min_component_area = max(0, min_component_area)
        if enabled is not None:
            self.enabled = enabled

    def __repr__(self) -> str:
        return (
            f"CardPreprocessor(enabled={self.enabled}, "
            f"lower_hsv={tuple(self.lower_hsv)}, upper_hsv={tuple(self.upper_hsv)}, "
            f"kernel_size={self.closing_kernel_size}, min_area={self.min_component_area})"
        )
