from .interface import ISegmenter
from .core import (
    HPP_Segmentation,
    VPP_Segmentation,
    CCA_Segmentation,
    Contour_Segmentation,
)
from src.dtypes import SegmentationType

from typing import Any, Dict


class SegmentationBuilder:
    @staticmethod
    def Build(type: SegmentationType, **kwargs: Dict[str, Any]) -> ISegmenter:
        if type == SegmentationType.HPP:
            min_height = kwargs.get("min_height", 5)
            margin = kwargs.get("margin", 2)
            threshold_ratio = kwargs.get("threshold_ratio", 0)
            return HPP_Segmentation(min_height, margin, threshold_ratio)

        if type == SegmentationType.VPP:
            min_width = kwargs.get("min_width", 3)
            margin = kwargs.get("margin", 2)
            threshold_ratio = kwargs.get("threshold_ratio", 0)
            return VPP_Segmentation(min_width, margin, threshold_ratio)

        if type == SegmentationType.CCA:
            min_height = kwargs.get("min_height", 5)
            return CCA_Segmentation(min_height)

        if type == SegmentationType.COUNTOUR:
            min_height = kwargs.get("min_height", 5)
            min_width = kwargs.get("min_width", 3)
            margin = kwargs.get("margin", 2)
            return Contour_Segmentation(min_height, min_width, margin)

        raise Exception("Invalid segmenter type")
