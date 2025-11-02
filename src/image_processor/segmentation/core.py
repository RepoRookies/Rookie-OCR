from .interface import ISegmenter
from src.image_processor.morphops import IMorphOperation

import cv2
import numpy as np
from typing import List


class HPP_Segmentation(ISegmenter):
    def __init__(self, min_height: int = 5, margin: int = 2,
                 threshold_ratio: float = 0.5, morphop: IMorphOperation = None):
        """
        Horizontal Projection Profile (HPP) Segmentation.
        Args:
            min_height (int): Minimum height for a segment to be considered.
            margin (int): Pixel margin added around the cropped line.
            threshold_ratio (float): Threshold relative to mean HPP value.
            morphop (IMorphOperation): Morphological preprocessing operation (optional).
        """
        self.min_height = min_height
        self.margin = margin
        self.threshold_ratio = threshold_ratio
        self.morphop = morphop

    def Segment(self, image: np.ndarray) -> List[np.ndarray]:
        if image.size == 0:
            return []

        processed_image = (self.morphop.Morph(image)
                           if self.morphop is not None else image)

        # Horizontal Projection Profile (HPP)
        hpp = np.sum(processed_image.astype(np.uint32), axis=1)
        mean_val = np.mean(hpp)
        threshold = mean_val * self.threshold_ratio

        start_indices, end_indices = [], []
        in_segment = False

        for i, val in enumerate(hpp):
            if val > threshold and not in_segment:
                start_indices.append(i)
                in_segment = True
            elif val <= threshold and in_segment:
                end_indices.append(i)
                in_segment = False
        if in_segment:
            end_indices.append(len(hpp))

        segments = []
        for start, end in zip(start_indices, end_indices):
            if end - start >= self.min_height:
                crop = image[max(0, start - self.margin):min(image.shape[0], end + self.margin), :]
                segments.append(crop)

        return segments


class VPP_Segmentation(ISegmenter):
    def __init__(self, min_width: int = 3, margin: int = 2,
                 threshold_ratio: float = 0.5, morphop: IMorphOperation = None):
        """
        Vertical Projection Profile (VPP) Segmentation.
        Args:
            min_width (int): Minimum width for a segment to be considered.
            margin (int): Pixel margin added around the cropped word.
            threshold_ratio (float): Threshold relative to mean VPP value.
            morphop (IMorphOperation): Morphological preprocessing operation (optional).
        """
        self.min_width = min_width
        self.margin = margin
        self.threshold_ratio = threshold_ratio
        self.morphop = morphop

    def Segment(self, image: np.ndarray) -> List[np.ndarray]:
        if image.size == 0:
            return []

        processed_image = (self.morphop.Morph(image)
                           if self.morphop is not None else image)

        # Vertical Projection Profile (VPP)
        vpp = np.sum(processed_image.astype(np.uint32), axis=0)
        vpp_smooth = cv2.blur(vpp.reshape(1, -1).astype(np.float32), (1, 5)).flatten()
        mean_val = np.mean(vpp_smooth)
        threshold = mean_val * self.threshold_ratio

        segments = []
        in_segment = False
        segment_start = 0

        for i, val in enumerate(vpp_smooth):
            if val > threshold and not in_segment:
                segment_start = i
                in_segment = True
            elif val <= threshold and in_segment:
                if i - segment_start >= self.min_width:
                    crop = image[:, max(0, segment_start - self.margin):min(image.shape[1], i + self.margin)]
                    segments.append(crop)
                in_segment = False

        if in_segment and len(vpp_smooth) - segment_start >= self.min_width:
            crop = image[:, max(0, segment_start - self.margin):min(image.shape[1], len(vpp_smooth) + self.margin)]
            segments.append(crop)

        return segments


class CCA_Segmentation(ISegmenter):
    def __init__(self, min_height: int = 5, morphop: IMorphOperation = None):
        """
        Connected Component Analysis (CCA) Segmentation.
        Args:
            min_height (int): Minimum height for a connected component to be considered.
            morphop (IMorphOperation): Morphological preprocessing operation (optional).
        """
        self.min_char_height = min_height
        self.morphop = morphop

    def Segment(self, image: np.ndarray) -> List[np.ndarray]:
        if image.size == 0:
            return []

        processed_image = (self.morphop.Morph(image)
                           if self.morphop is not None else image)

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            processed_image.astype(np.uint8), 8, cv2.CV_32S
        )

        components = []
        for i in range(1, num_labels):  # skip background
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                         stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]

            if h >= self.min_char_height:
                crop = image[y:y + h, x:x + w]
                components.append((x, crop))

        components.sort(key=lambda item: item[0])
        return [comp[1] for comp in components]


class Contour_Segmentation(ISegmenter):
    def __init__(self, min_height: int = 5, min_width: int = 3,
                 margin: int = 2, morphop: IMorphOperation = None):
        """
        Contour-based segmentation.
        Args:
            min_height (int): Minimum height for a contour to be considered.
            min_width (int): Minimum width for a contour to be considered.
            margin (int): Pixel margin around the cropped contour.
            morphop (IMorphOperation): Morphological preprocessing operation (optional).
        """
        self.min_height = min_height
        self.min_width = min_width
        self.margin = margin
        self.morphop = morphop

    def Segment(self, image: np.ndarray) -> List[np.ndarray]:
        if image.size == 0:
            return []

        processed_image = (self.morphop.Morph(image)
                           if self.morphop is not None else image)

        contours, _ = cv2.findContours(
            processed_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        segments = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h >= self.min_height and w >= self.min_width:
                crop = image[max(0, y - self.margin):min(image.shape[0], y + h + self.margin),
                             max(0, x - self.margin):min(image.shape[1], x + w + self.margin)]
                segments.append((x, crop))

        segments.sort(key=lambda s: s[0])
        return [s[1] for s in segments]
