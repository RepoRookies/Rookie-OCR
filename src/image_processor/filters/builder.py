from .interface import IFilter
from .core import (
    AverageFilter,
    MedianFilter,
    GaussianFilter,
    SobelFilter,
    LaplacianFilter,
    UnsharpMaskingFilter,
    HighBoostFilter,
)
from src.dtypes import FilterType

from typing import Any, Dict


class FilterBuilder:
    @staticmethod
    def Build(type: FilterType, **kwargs: Dict[str, Any]) -> IFilter:
        if type == FilterType.AVERAGE:
            kernel_size = kwargs.get("kernel_size", 3)
            return AverageFilter(kernel_size)

        if type == FilterType.MEDIAN:
            kernel_size = kwargs.get("kernel_size", 3)
            return MedianFilter(kernel_size)

        if type == FilterType.GAUSSIAN:
            sigma = kwargs.get("sigma", 1.0)
            kernel_size = kwargs.get("kernel_size", None)
            return GaussianFilter(sigma, kernel_size)

        if type == FilterType.SOBEL:
            axis = kwargs.get("axis", 0)
            return SobelFilter(axis)

        if type == FilterType.LAPLACIAN:
            return LaplacianFilter()

        if type == FilterType.UNSHARP_MASKING:
            sigma = kwargs.get("sigma", 1.0)
            strength = kwargs.get("strength", 1.0)
            return UnsharpMaskingFilter(sigma, strength)

        if type == FilterType.HIGH_BOOST:
            sigma = kwargs.get("sigma", 1.0)
            A = kwargs.get("A", 1.0)
            return HighBoostFilter(sigma, A)

        raise ValueError("Invalid filter type")
