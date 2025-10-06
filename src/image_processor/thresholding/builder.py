from .interface import IThresholding
from .core import (
    GlobalThresholding,
    AdaptiveMeanThresholding,
    AdaptiveGaussianThresholding,
    OtsuThresholding,
)
from src.dtypes import ThresholdingMode, ThresholdingType

from typing import Any, Dict


class ThresholdingBuilder:
    @staticmethod
    def Build(
        type: ThresholdingType, mode: ThresholdingMode, **kwargs: Dict[str, Any]
    ) -> IThresholding:
        max_value = kwargs.get("max_value", 255.0)

        if type == ThresholdingType.GLOBAL:
            threshold = kwargs.get("threshold", 127.0)
            return GlobalThresholding(mode, threshold, max_value)

        if type == ThresholdingType.ADAPTIVE_MEAN:
            block_size = kwargs.get("block_size", 11)
            C = kwargs.get("C", 3.0)
            return AdaptiveMeanThresholding(mode, block_size, C, max_value)

        if type == ThresholdingType.ADAPTIVE_GAUSSIAN:
            sigma = kwargs.get("sigma", 1.0)
            block_size = kwargs.get("block_size", 11)
            C = kwargs.get("C", 3.0)
            return AdaptiveGaussianThresholding(mode, block_size, sigma, C, max_value)

        if type == ThresholdingType.OTSU:
            return OtsuThresholding(mode, max_value)

        raise ValueError("Invalid thresholding type")
