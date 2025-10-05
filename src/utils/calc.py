import cv2
import numpy as np


class Calc:
    @staticmethod
    def Convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Applies a convolution to an image using a given kernel
        Args:
            image (np.ndarray, 2D): The input image
            kernel (np.ndarray, 2D): The convolution kernel
        Returns:
            image (np.ndarray, 2D): The convolved image
        """
        flipped_kernel = cv2.flip(kernel, -1)
        return cv2.filter2D(image, -1, flipped_kernel)

    @staticmethod
    def Correlate(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Applies a correlation to an image using a given kernel
        Args:
            image (np.ndarray, 2D): The input image
            kernel (np.ndarray, 2D): The correlation kernel
        Returns:
            image (np.ndarray, 2D): The correlated image
        """
        return cv2.filter2D(image, -1, kernel)