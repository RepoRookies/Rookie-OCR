import numpy as np


class IFilter:
    def __init__(self):
        pass

    def Filter(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError
