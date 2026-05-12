"""
UNO Card Detection Pipeline using SIFT Feature Matching and Homography.

This module implements an academic approach to card recognition by:
1. Extracting SIFT keypoints and descriptors from reference card images
2. Matching features between test images and reference cards using BFMatcher
3. Filtering ambiguous matches using Lowe's ratio test
4. Validating matches using RANSAC-based homography estimation
5. Transforming reference bounding boxes using the computed homography
6. Visualizing detected cards with labeled bounding boxes

No deep learning or pre-trained models are used (academic requirements).
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from card_preprocessor import CardPreprocessor


@dataclass
class CardDetection:
    """Data class representing a single card detection."""

    label: str
    bbox: Tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max)
    corners: np.ndarray  # (4, 2) array of warped corners
    confidence: float  # Combined score [0, 1]
    num_matches: int  # Number of matched keypoints
    num_inliers: int  # Number of RANSAC inliers


class UNOCardDetector:
    """
    SIFT-based detector for UNO cards using homography transformation.

    Workflow:
    ---------
    1. Load reference card images from a directory (PNG files)
    2. Extract and cache SIFT descriptors for each reference
    3. For each test image:
        a. Extract SIFT features
        b. Match against all references using BFMatcher
        c. Filter using Lowe's ratio test (threshold ≈ 0.75)
        d. Validate matches with RANSAC homography
        e. Transform reference bounding box
        f. Collect confident detections
    4. Visualize with matplotlib
    """

    def __init__(
        self,
        reference_dir: str,
        lowe_ratio: float = 0.75,
        min_matches: int = 3,
        ransac_threshold: float = 5.0,
        ransac_confidence: float = 0.99,
        min_confidence: float = 0.0,
        max_detections_per_ref: int = 1,
        allow_multiple_instances: bool = True,
        nms_threshold: float = 0.3,
        expected_size: Tuple[int, int] = (450, 450),
        size_tolerance: float = 0.2,
    ):
        """
        Initialize the detector.

        Parameters
        ----------
        reference_dir : str
            Path to directory containing reference PNG images.
            Filenames are parsed as labels (e.g., "b_5.png" → label "b_5")
        lowe_ratio : float
            Lowe's ratio test threshold (default 0.75).
            Ratio of distance to 2nd NN vs best NN. Lower = more strict.
        min_matches : int
            Minimum number of matched keypoints for homography (default 4).
        ransac_threshold : float
            RANSAC reprojection error threshold in pixels (default 5.0).
        ransac_confidence : float
            RANSAC confidence level (default 0.99).
        """
        self.reference_dir = Path(reference_dir)
        self.lowe_ratio = lowe_ratio
        self.min_matches = min_matches
        self.ransac_threshold = ransac_threshold
        self.ransac_confidence = ransac_confidence

        # Initialize SIFT detector
        self.sift = cv2.SIFT_create()

        # Storage for reference images and their SIFT features
        self.references: Dict[str, Dict] = {}

        # Detection behavior
        self.min_confidence = min_confidence
        self.max_detections_per_ref = max_detections_per_ref
        self.allow_multiple_instances = allow_multiple_instances
        self.debug = False
        self.nms_threshold = nms_threshold
        # Expected object size (width, height) in pixels and tolerance fraction
        self.expected_size = expected_size
        self.size_tolerance = size_tolerance

        # Initialize preprocessing (disabled by default)
        self.preprocessor = CardPreprocessor(enabled=False)

        # Load reference images
        self._load_references()

    def set_preprocessing(
        self,
        enabled: bool = False,
        lower_hsv: Optional[Tuple[int, int, int]] = None,
        upper_hsv: Optional[Tuple[int, int, int]] = None,
        closing_kernel_size: Optional[int] = None,
        min_component_area: Optional[int] = None,
    ) -> None:
        """
        Configure preprocessing parameters.

        Parameters
        ----------
        enabled : bool
            Enable or disable preprocessing.
        lower_hsv : Tuple[int, int, int], optional
            Lower HSV threshold (H: 0-179, S/V: 0-255).
        upper_hsv : Tuple[int, int, int], optional
            Upper HSV threshold.
        closing_kernel_size : int, optional
            Kernel size for morphological closing.
        min_component_area : int, optional
            Minimum area to keep a connected component.
        """
        self.preprocessor.set_parameters(
            enabled=enabled,
            lower_hsv=lower_hsv,
            upper_hsv=upper_hsv,
            closing_kernel_size=closing_kernel_size,
            min_component_area=min_component_area,
        )

    def _nms(
        self, detections: List[CardDetection], iou_threshold: float = 0.3
    ) -> List[CardDetection]:
        """
        Apply Non-Maximum Suppression to remove overlapping detections.
        Keeps detections with highest confidence.
        """
        if not detections:
            return detections

        # Sort by confidence (descending)
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        keep = []

        for det in sorted_dets:
            x1_a, y1_a, x2_a, y2_a = det.bbox
            should_keep = True

            for kept_det in keep:
                x1_b, y1_b, x2_b, y2_b = kept_det.bbox

                # Compute IoU
                xi1 = max(x1_a, x1_b)
                yi1 = max(y1_a, y1_b)
                xi2 = min(x2_a, x2_b)
                yi2 = min(y2_a, y2_b)

                inter_w = max(0, xi2 - xi1)
                inter_h = max(0, yi2 - yi1)
                inter_area = inter_w * inter_h

                area_a = (x2_a - x1_a) * (y2_a - y1_a)
                area_b = (x2_b - x1_b) * (y2_b - y1_b)
                union_area = area_a + area_b - inter_area

                if union_area > 0:
                    iou = inter_area / union_area
                    if iou > iou_threshold:
                        should_keep = False
                        break

            if should_keep:
                keep.append(det)

        return keep

    def _load_references(self) -> None:
        """
        Load reference PNG images and pre-compute SIFT descriptors.

        Expected structure:
        - reference_dir/*.png
        - Filenames are used as labels
        """
        print(f"Loading reference images from {self.reference_dir}")

        png_files = sorted(self.reference_dir.glob("*.png"))

        if not png_files:
            raise ValueError(f"No PNG files found in {self.reference_dir}")

        for png_path in png_files:
            label = png_path.stem  # Filename without extension

            # Load image in grayscale
            img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  ⚠ Failed to load {png_path}")
                continue

            # Also load in color for visualization
            img_color = cv2.imread(str(png_path), cv2.IMREAD_COLOR)

            # Extract SIFT features
            kp, des = self.sift.detectAndCompute(img, None)

            # Store reference data
            self.references[label] = {
                "image_gray": img,
                "image_color": img_color,
                "keypoints": kp,
                "descriptors": des,
                "shape": img.shape,  # (height, width)
            }

            num_kp = len(kp) if kp is not None else 0
            print(f"  Loaded {label:20s} | Shape: {img.shape} | Keypoints: {num_kp}")

        print(f"\n  Total references loaded: {len(self.references)}\n")

    def extract_sift(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        Extract SIFT keypoints and descriptors from an image.

        Parameters
        ----------
        image : np.ndarray
            Input image (assumed to be grayscale or will be converted).

        Returns
        -------
        keypoints : List
            SIFT keypoints (cv2.KeyPoint objects)
        descriptors : np.ndarray
            SIFT descriptors (N, 128) array
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        kp, des = self.sift.detectAndCompute(image, None)

        return kp, des

    def _lowe_ratio_test(
        self, matches: List, descriptors_test: np.ndarray, descriptors_ref: np.ndarray
    ) -> Tuple[List, np.ndarray]:
        """
        Apply Lowe's ratio test to filter ambiguous matches.

        The ratio test compares the distance to the nearest neighbor vs.
        the distance to the second nearest neighbor. If the ratio is < lowe_ratio,
        the match is considered reliable.

        Parameters
        ----------
        matches : List
            List of DMatch objects (assumes knnMatch with k=2)
        descriptors_test : np.ndarray
            Test image descriptors
        descriptors_ref : np.ndarray
            Reference image descriptors

        Returns
        -------
        good_matches : List
            Filtered matches passing the ratio test
        confidence_scores : np.ndarray
            Confidence for each good match (1 - ratio)
        """
        good_matches = []
        confidence_scores = []

        for match_pair in matches:
            # match_pair is a list of 2 DMatch objects
            if len(match_pair) != 2:
                continue

            m, n = match_pair
            ratio = m.distance / n.distance

            # Apply ratio test
            if ratio < self.lowe_ratio:
                good_matches.append(m)
                # Confidence inversely related to ratio (higher ratio = lower confidence)
                confidence = 1.0 - ratio
                confidence_scores.append(confidence)

        return good_matches, (
            np.array(confidence_scores) if confidence_scores else np.array([])
        )

    def _estimate_homography(
        self, keypoints_test: List, keypoints_ref: List, matches: List
    ) -> Tuple[Optional[np.ndarray], int, Optional[np.ndarray]]:
        """
        Estimate homography using RANSAC and filter inliers.

        Parameters
        ----------
        keypoints_test : List
            Keypoints from test image (cv2.KeyPoint objects)
        keypoints_ref : List
            Keypoints from reference image (cv2.KeyPoint objects)
        matches : List
            DMatch objects connecting test to reference keypoints

        Returns
        -------
        homography : np.ndarray or None
            3x3 homography matrix, or None if estimation fails
        num_inliers : int
            Number of inlier matches used to compute homography
        """
        if len(matches) < self.min_matches:
            return None, 0

        # Extract coordinates of matched keypoints
        src_pts = np.float32([keypoints_test[m.queryIdx].pt for m in matches]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([keypoints_ref[m.trainIdx].pt for m in matches]).reshape(
            -1, 1, 2
        )

        # Compute homography using RANSAC
        # Compute homography mapping test -> ref (src_pts -> dst_pts)
        H, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            self.ransac_threshold,
            confidence=self.ransac_confidence,
        )

        if H is None:
            return None, 0, None

        # Count inliers
        num_inliers = int(np.sum(mask)) if mask is not None else 0

        # Return mask as flattened boolean array for identifying inliers
        mask_flat = mask.reshape(-1) if mask is not None else None

        # findHomography was computed with src=test -> dst=ref. We need a
        # homography mapping reference -> test to warp reference corners into
        # the test image. Compute the inverse if possible.
        try:
            H_inv = np.linalg.inv(H)
        except Exception:
            H_inv = None

        return H_inv, num_inliers, mask_flat

    def _warp_bounding_box(
        self, homography: np.ndarray, ref_shape: Tuple[int, int]
    ) -> Tuple[Tuple[float, float, float, float], np.ndarray]:
        """
        Warp the reference image's bounding box using the homography.

        The reference image's bounding box is defined by its four corners.
        These corners are transformed by the homography to get the warped corners
        in the test image space. Then we compute the axis-aligned bounding box.

        Parameters
        ----------
        homography : np.ndarray
            3x3 homography matrix (from ref to test)
        ref_shape : Tuple[int, int]
            Shape of reference image (height, width)

        Returns
        -------
        bbox : Tuple[float, float, float, float]
            Axis-aligned bounding box (x_min, y_min, x_max, y_max) in test image
        corners_warped : np.ndarray
            Warped corners (4, 2) of the reference image
        """
        height, width = ref_shape

        # Define the four corners of the reference image
        corners = np.array(
            [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
        ).reshape(-1, 1, 2)

        # Apply homography transformation
        corners_warped = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)

        # Compute axis-aligned bounding box
        x_min = np.min(corners_warped[:, 0])
        x_max = np.max(corners_warped[:, 0])
        y_min = np.min(corners_warped[:, 1])
        y_max = np.max(corners_warped[:, 1])

        return (x_min, y_min, x_max, y_max), corners_warped

    def match_card(
        self, test_image: np.ndarray, kp_test: List, des_test: np.ndarray, label: str
    ) -> List[Dict]:
        """
        Match a test image against a single reference card.

        Parameters
        ----------
        test_image : np.ndarray
            Test image
        kp_test : List
            Test image keypoints
        des_test : np.ndarray
            Test image descriptors
        label : str
            Reference card label

        Returns
        -------
        detection : Dict or None
            Detection dictionary with label, bbox, corners, confidence, etc.
            Returns None if match is not confident enough.
        """
        if label not in self.references:
            return []

        ref_data = self.references[label]
        kp_ref = ref_data["keypoints"]
        des_ref = ref_data["descriptors"]

        # Handle case where reference has no descriptors
        if kp_ref is None or des_ref is None or len(kp_ref) == 0:
            return None

        # Handle case where test has no descriptors
        if des_test is None or len(kp_test) == 0:
            return None

        # Create BFMatcher (for SIFT descriptors, use L2 norm)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        # knnMatch to get 2 nearest neighbors
        try:
            matches = bf.knnMatch(des_test, des_ref, k=2)
        except Exception:
            # Descriptor dimension mismatch or other issue
            matches = []

        good_matches, conf_scores = [], np.array([])
        if matches and len(matches) > 0:
            # Apply Lowe's ratio test
            good_matches, conf_scores = self._lowe_ratio_test(
                matches, des_test, des_ref
            )

        # Fallback: if not enough good matches, try cross-checked matching
        if len(good_matches) < self.min_matches:
            bf_cc = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            try:
                matches_cc = bf_cc.match(des_test, des_ref)
            except Exception:
                matches_cc = []

            if matches_cc and len(matches_cc) >= self.min_matches:
                # Sort by distance (lower is better)
                matches_cc = sorted(matches_cc, key=lambda m: m.distance)
                # Use top matches as good_matches
                good_matches = matches_cc[
                    : max(self.min_matches, int(len(matches_cc) * 0.8))
                ]
                # Create confidence scores by normalizing distances
                distances = np.array([m.distance for m in good_matches], dtype=float)
                maxd = distances.max() if distances.size else 1.0
                conf_scores = 1.0 - (distances / (maxd + 1e-8))

        if len(good_matches) < self.min_matches:
            if self.debug:
                print(f"  [DEBUG] {label}: insufficient matches ({len(good_matches)})")
            return []

        detections = []

        # If multiple instances allowed, iteratively estimate homography,
        # collect inliers, remove them, and repeat to find other instances.
        remaining_matches = good_matches.copy()
        remaining_conf = conf_scores.copy() if conf_scores.size else np.array([])

        iter_count = 0
        while len(remaining_matches) >= self.min_matches:
            H, num_inliers, mask_flat = self._estimate_homography(
                kp_test, kp_ref, remaining_matches
            )

            if H is None or num_inliers < self.min_matches:
                break

            # Identify inlier matches
            if mask_flat is None:
                break

            mask_bool = mask_flat.astype(bool)
            inlier_matches = [
                m for m, flag in zip(remaining_matches, mask_bool) if flag
            ]
            inlier_conf = (
                remaining_conf[mask_bool] if remaining_conf.size else np.array([])
            )

            if len(inlier_matches) < self.min_matches:
                break

            # Warp bounding box
            bbox, corners_warped = self._warp_bounding_box(H, ref_data["shape"])

            # Compute combined confidence score
            conf_matches = float(np.mean(inlier_conf)) if inlier_conf.size else 0.0
            conf_ransac = (
                float(len(inlier_matches)) / float(len(remaining_matches))
                if len(remaining_matches) > 0
                else 0.0
            )
            confidence = 0.5 * conf_matches + 0.5 * conf_ransac

            detection = {
                "label": label,
                "bbox": bbox,
                "corners": corners_warped,
                "confidence": confidence,
                "num_matches": len(inlier_matches),
                "num_inliers": int(num_inliers),
            }

            # Accept detection only if above minimum confidence
            if confidence >= self.min_confidence:
                detections.append(detection)

            iter_count += 1
            if (not self.allow_multiple_instances) or (
                iter_count >= self.max_detections_per_ref
            ):
                break

            # Remove inlier matches from remaining and continue searching
            remaining_matches = [
                m for m, flag in zip(remaining_matches, ~mask_bool) if flag
            ]
            if remaining_conf.size:
                remaining_conf = remaining_conf[~mask_bool]

        return detections

    def detect(self, test_image: np.ndarray) -> List[CardDetection]:
        """
        Detect all UNO cards in a test image.

        Parameters
        ----------
        test_image : np.ndarray
            Input test image (BGR or RGB)

        Returns
        -------
        detections : List[CardDetection]
            List of detected cards, sorted by confidence (descending).
        """
        # Apply optional preprocessing (HSV thresholding + morphology)
        test_image_processed, _ = self.preprocessor.preprocess(test_image)

        # Convert to grayscale for SIFT
        if len(test_image_processed.shape) == 3:
            gray = cv2.cvtColor(test_image_processed, cv2.COLOR_BGR2GRAY)
        else:
            gray = test_image_processed

        # Extract SIFT features from test image
        kp_test, des_test = self.extract_sift(gray)

        if des_test is None or len(kp_test) == 0:
            print("  ⚠ No SIFT features found in test image")
            return []

        detections = []

        # Match against all reference cards
        for label in self.references:
            det_list = self.match_card(test_image, kp_test, des_test, label)
            for det in det_list:
                detections.append(CardDetection(**det))

        # Filter out detections with invalid or extreme bounding boxes
        img_h, img_w = test_image.shape[0], test_image.shape[1]
        filtered = []
        for d in detections:
            x_min, y_min, x_max, y_max = d.bbox
            # Check for NaN/inf
            if not np.isfinite(x_min + y_min + x_max + y_max):
                continue
            # Normalize and clamp
            w = x_max - x_min
            h = y_max - y_min
            if w <= 5 or h <= 5:
                continue
            # Reject boxes that are absurdly large or completely outside image
            area = w * h
            if area <= 0 or area > (img_w * img_h * 1.2):
                continue
            # Reject boxes far outside image bounds
            if x_max < -0.5 * img_w or x_min > 1.5 * img_w:
                continue
            if y_max < -0.5 * img_h or y_min > 1.5 * img_h:
                continue
            # Size-based filter (expected size with tolerance)
            exp_w, exp_h = self.expected_size
            tol = self.size_tolerance
            min_w = exp_w * (1.0 - tol)
            max_w = exp_w * (1.0 + tol)
            min_h = exp_h * (1.0 - tol)
            max_h = exp_h * (1.0 + tol)
            # Accept detection only if width and height are roughly in expected range
            if not (min_w <= w <= max_w and min_h <= h <= max_h):
                continue
            filtered.append(d)

        # Apply NMS to remove overlapping detections
        nms_filtered = self._nms(filtered, iou_threshold=self.nms_threshold)

        return nms_filtered

    def visualize(
        self,
        test_image: np.ndarray,
        detections: List[CardDetection],
        show_confidence: bool = True,
        figsize: Tuple[int, int] = (14, 10),
    ) -> plt.Figure:
        """
        Visualize detected cards with bounding boxes and labels.

        Parameters
        ----------
        test_image : np.ndarray
            Input test image (BGR)
        detections : List[CardDetection]
            List of detected cards
        show_confidence : bool
            Whether to display confidence scores (default True)
        figsize : Tuple[int, int]
            Figure size (width, height) in inches

        Returns
        -------
        fig : plt.Figure
            Matplotlib figure object
        """
        # Convert BGR to RGB for matplotlib
        if len(test_image.shape) == 3 and test_image.shape[2] == 3:
            img_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = test_image

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(img_rgb)
        ax.set_title(
            f"UNO Card Detection: {len(detections)} cards found",
            fontsize=14,
            fontweight="bold",
        )

        # Color palette for boxes
        colors = plt.cm.tab20(np.linspace(0, 1, len(detections)))

        for idx, detection in enumerate(detections):
            x_min, y_min, x_max, y_max = detection.bbox

            # Ensure coordinates are non-negative and within image bounds
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(test_image.shape[1], x_max)
            y_max = min(test_image.shape[0], y_max)

            width = x_max - x_min
            height = y_max - y_min

            # Draw rectangle
            color = colors[idx % len(colors)]
            rect = patches.Rectangle(
                (x_min, y_min),
                width,
                height,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)

            # Create label text
            if show_confidence:
                label_text = f"{detection.label}\n{detection.confidence:.2f}"
            else:
                label_text = detection.label

            # Draw label background and text
            text_y = max(y_min - 10, 20)
            ax.text(
                x_min,
                text_y,
                label_text,
                fontsize=9,
                fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
            )

        ax.axis("off")
        plt.tight_layout()

        return fig

    def process_image(
        self, image_path: str, display: bool = True
    ) -> Tuple[np.ndarray, List[CardDetection]]:
        """
        Process a single image: load, detect, and optionally visualize.

        Parameters
        ----------
        image_path : str
            Path to test image
        display : bool
            Whether to display the result (default True)

        Returns
        -------
        test_image : np.ndarray
            Loaded test image
        detections : List[CardDetection]
            List of detected cards
        """
        print(f"\nProcessing: {image_path}")

        # Load image
        test_image = cv2.imread(image_path)
        if test_image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        print(f"  Image shape: {test_image.shape}")

        # Detect
        detections = self.detect(test_image)

        print(f"  Detections: {len(detections)}")
        for det in detections:
            print(
                f"    - {det.label}: confidence={det.confidence:.3f}, "
                + f"matches={det.num_matches}, inliers={det.num_inliers}"
            )

        # Visualize
        if display:
            self.visualize(test_image, detections)
            plt.show()

        return test_image, detections
