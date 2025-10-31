import numpy as np
from abc import ABC, abstractmethod

class IInterpolator(ABC):
    @abstractmethod
    def Interpolate(self, image: np.ndarray) -> np.ndarray:
        """
        Abstract interpolation method to be implemented by all interpolation operations.
        Args:
            image (np.ndarray): Input image to be interpolated.
        Returns:
            np.ndarray: Interpolated (resized) output image.
        """
        raise NotImplementedError
