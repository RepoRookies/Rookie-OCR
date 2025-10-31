import numpy as np
from abc import ABC, abstractmethod


class IMorphOperation(ABC):
    @abstractmethod
    def Morph(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError

