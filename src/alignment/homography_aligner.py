"""
Homography-based image alignment using OpenCV ORB features.

Aligns a *source* label image to a *reference* template image so that
downstream ROI-based size extraction uses consistent pixel coordinates.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Minimum number of good matches required to attempt homography estimation
_MIN_MATCH_COUNT = 10


class HomographyAligner:
    """
    Align a source image to a reference image using ORB keypoints and a
    homography computed from matched feature pairs.

    Parameters
    ----------
    max_features : int
        Maximum number of ORB keypoints to detect per image (default 500).
    good_match_ratio : float
        Lowe's ratio-test threshold: a match is *good* when
        ``distance_1 / distance_2 < good_match_ratio`` (default 0.75).
    """

    def __init__(self, max_features: int = 500, good_match_ratio: float = 0.75) -> None:
        self.max_features = max_features
        self.good_match_ratio = good_match_ratio

        self.orb: cv2.ORB = cv2.ORB_create(max_features)
        # BFMatcher with Hamming distance for binary ORB descriptors
        self.matcher: cv2.BFMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def align(
        self,
        source: np.ndarray,
        reference: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """
        Warp *source* so it is geometrically aligned to *reference*.

        Parameters
        ----------
        source : np.ndarray
            The image to align (BGR or grayscale).
        reference : np.ndarray
            The template to align against (BGR or grayscale).

        Returns
        -------
        (aligned_image, success) : tuple[np.ndarray, bool]
            On success, ``aligned_image`` is the warped source, same size as
            ``reference``.  On failure, the original ``source`` is returned
            unchanged and ``success`` is ``False``.
        """
        src_gray = self._to_gray(source)
        ref_gray = self._to_gray(reference)

        kp_src, des_src = self.orb.detectAndCompute(src_gray, None)
        kp_ref, des_ref = self.orb.detectAndCompute(ref_gray, None)

        if des_src is None or des_ref is None:
            logger.warning("ORB: no descriptors found (src=%d ref=%d kps).",
                           len(kp_src) if kp_src else 0, len(kp_ref) if kp_ref else 0)
            return source, False

        good_matches = self._ratio_test(kp_src, kp_ref, des_src, des_ref)
        logger.debug("ORB good matches: %d (need >= %d)", len(good_matches), _MIN_MATCH_COUNT)

        if len(good_matches) < _MIN_MATCH_COUNT:
            logger.info(
                "Alignment skipped: only %d good matches (threshold %d).",
                len(good_matches), _MIN_MATCH_COUNT,
            )
            return source, False

        # Build point correspondence arrays
        src_pts = np.float32([kp_src[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        ref_pts = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, ref_pts, cv2.RANSAC, ransacReprojThreshold=5.0)
        if H is None:
            logger.warning("Homography estimation failed — returning source unchanged.")
            return source, False

        h_ref, w_ref = reference.shape[:2]
        aligned = cv2.warpPerspective(source, H, (w_ref, h_ref),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE)
        inliers = int(mask.ravel().sum()) if mask is not None else len(good_matches)
        logger.info(
            "Alignment succeeded: %d inliers / %d good matches.",
            inliers, len(good_matches),
        )
        return aligned, True

    def estimate_quality(self, source: np.ndarray, reference: np.ndarray) -> float:
        """
        Estimate the alignment quality between two images.

        The score is computed as the fraction of *good* feature matches relative
        to the maximum number of detectable keypoints.  It reflects how
        structurally similar the two images are, not actual warp accuracy.

        Parameters
        ----------
        source : np.ndarray
        reference : np.ndarray

        Returns
        -------
        float
            Quality score in [0, 1].  Values above ~0.3 generally indicate a
            reliable homography; below 0.1 means the images are likely unrelated.
        """
        src_gray = self._to_gray(source)
        ref_gray = self._to_gray(reference)

        kp_src, des_src = self.orb.detectAndCompute(src_gray, None)
        kp_ref, des_ref = self.orb.detectAndCompute(ref_gray, None)

        if des_src is None or des_ref is None or len(kp_src) == 0 or len(kp_ref) == 0:
            return 0.0

        good_matches = self._ratio_test(kp_src, kp_ref, des_src, des_ref)
        # Normalise by the smaller of the two keypoint counts
        max_possible = min(len(kp_src), len(kp_ref), self.max_features)
        if max_possible == 0:
            return 0.0

        score = min(1.0, len(good_matches) / max_possible)
        logger.debug("Alignment quality score: %.3f (%d good / %d max)",
                     score, len(good_matches), max_possible)
        return score

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ratio_test(
        self,
        kp_src: list,
        kp_ref: list,
        des_src: np.ndarray,
        des_ref: np.ndarray,
    ) -> list:
        """Apply Lowe's ratio test and return only good matches."""
        # knnMatch with k=2 for ratio test
        raw_matches = self.matcher.knnMatch(des_src, des_ref, k=2)
        good: list = []
        for pair in raw_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.good_match_ratio * n.distance:
                good.append(m)
        return good

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        """Convert BGR image to grayscale; pass through if already single-channel."""
        if image.ndim == 2:
            return image
        if image.shape[2] == 1:
            return image[:, :, 0]
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
