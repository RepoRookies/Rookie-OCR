from .interface import IMorphOperation
from .core import Dilator, Eroder, Opener, Closer
from src.dtypes import MorphOperationType

from typing import Any, Dict


class MorphOperationBuilder:
    @staticmethod
    def Build(type: MorphOperationType, **kwargs: Dict[str, Any]) -> IMorphOperation:
        kernel = kwargs.get("kernel", None)

        if type == MorphOperationType.DILATION:
            return Dilator(kernel)

        if type == MorphOperationType.EROSION:
            return Eroder(kernel)

        if type == MorphOperationType.OPENING:
            return Opener(kernel)

        if type == MorphOperationType.CLOSING:
            return Closer(kernel)

        raise ValueError("Invalid morphological operation type")
