import numpy as np
from abc import ABC, abstractmethod


class IThresholding(ABC):
    @abstractmethod
    def ApplyThresholding(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError
