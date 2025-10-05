import numpy as np
from abc import ABC, abstractmethod


class IFilter(ABC):
    @abstractmethod
    def Filter(self, image: np.ndarray) -> np.ndarray:
        pass
