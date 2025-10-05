from src.filters.interface import IFilter
from src.filters.core import GaussianFilter
from src.types.filters import FilterType

from typing import Any, Dict


class FilterBuilder:
    @staticmethod
    def Build(type: FilterType, **kwargs: Dict[str, Any]) -> IFilter:
        if type == FilterType.GAUSSIAN:
            sigma = kwargs.get("sigma", 1.0)
            kernel_size = kwargs.get("kernel_size", None)
            return GaussianFilter(sigma, kernel_size)

        # if type == FilterType.MEDIAN:
        #     return MedianFilter(**kwargs)

        # if type == FilterType.SOBEL:
        #     return SobelFilter(**kwargs)

        # if type == FilterType.LAPLACIAN:
        #     return LaplacianFilter(**kwargs)

        raise Exception("Invalid filter type")
