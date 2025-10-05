import numpy as np
from typing import List

class ISegmenter:
    def __init__(self):
        pass

    def Segment(self, image: np.ndarray) -> List[np.ndarray]:
        raise NotImplementedError