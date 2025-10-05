from src.filters.interface import IFilter

import cv2
import numpy as np


class GaussianFilter(IFilter):
    def __init__(self, sigma: float = 1.0, kernel_size: int = None):
        """Constructor for GaussianFilter class
        Args:
            sigma (float, optional): Standard deviation of the Gaussian kernel. Defaults to 1.0.
            kernel_size (int, optional): Size of the Gaussian kernel. Defaults to None.
        """
        self.sigma = sigma
        if kernel_size is None:
            kernel_size = int(2 * self.sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.kernel = self.GenerateKernel()

    def GenerateKernel(self) -> np.ndarray:
        """Generates the Gaussian kernel
        Returns:
            kernel (np.ndarray, 2D): The Gaussian kernel
        """
        ax = np.arange(-self.kernel_size // 2 + 1.0, self.kernel_size // 2 + 1.0)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2.0 * self.sigma**2))
        kernel /= np.sum(kernel)
        return kernel

    def Filter(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur to an image using convolution
        Args:
            image (np.ndarray, 2D): The input image to be filtered
        Returns:
            image (np.ndarray, 2D): The filtered image
        """
        return cv2.filter2D(image, -1, self.kernel)
