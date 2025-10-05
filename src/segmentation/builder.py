from src.segmentation.interface import ISegmenter 
from src.segmentation.core import HPP_Segmentation, VPP_Segmentation, CCA_Segmentation 
from src.types.segmentation import SegmentationType

from typing import Any, Dict


class SegmentationBuilder:
    @staticmethod
    def Build(type: SegmentationType, **kwargs: Dict[str, Any]) -> ISegmenter:
        if type == SegmentationType.HPP:
            min_line_height = kwargs.get("min_height", 5)
            margin = kwargs.get("margin", 2)
            return HPP_Segmentation(min_line_height, margin)

        if type == SegmentationType.VPP:
            min_word_width = kwargs.get("min_width", 3)
            margin = kwargs.get("margin", 2)
            return VPP_Segmentation(min_word_width, margin)

        if type == SegmentationType.CCA:
            min_char_height = kwargs.get("min_height", 5)
            return CCA_Segmentation(min_char_height)

        raise Exception("Invalid segmenter type")