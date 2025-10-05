import numpy as np
from typing import List
from abc import ABC, abstractmethod


class ISegmenter(ABC):
    @abstractmethod
    def Segment(self, image: np.ndarray) -> List[np.ndarray]:
        raise NotImplementedError
