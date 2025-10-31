
from typing import Any, Dict
from .interface import IInterpolator
from .core import Upscaler, Downscaler, Resizer
from src.dtypes import InterpolationOperationType


class InterpolatorBuilder:
    @staticmethod
    def Build(type: InterpolationOperationType, **kwargs: Dict[str, Any]) -> IInterpolator:
        """
        Factory method to build specific interpolation operations based on type.
        Args:
            type (InterpolationOperationType): Type of interpolation operation to create.
            **kwargs: Parameters specific to the chosen interpolation operation.
        Returns:
            IInterpolator: Instance of a concrete interpolation class.
        """
        if type == InterpolationOperationType.UPSCALE:
            return Upscaler(
                scale_factor=kwargs.get("scale_factor", 2.0),
                interpolation=kwargs.get("interpolation")
            )

        if type == InterpolationOperationType.DOWNSCALE:
            return Downscaler(
                scale_factor=kwargs.get("scale_factor", 0.5),
                interpolation=kwargs.get("interpolation")
            )

        if type == InterpolationOperationType.RESIZE:
            return Resizer(
                target_size=kwargs.get("target_size"),
                interpolation=kwargs.get("interpolation")
            )

        raise ValueError("Invalid interpolation operation type")