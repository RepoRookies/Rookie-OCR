from .interface import IFilter
from src.utils import CVMath

import numpy as np


class AverageFilter(IFilter):
    def __init__(self, kernel_size: int = 5):
        """Constructor for AverageFilter class
        Args:
            kernel_size (int, optional): Size of the average filter kernel. Defaults to 5.
        """
        self.kernel_size = kernel_size
        self.kernel = self.GenerateKernel()

    def GenerateKernel(self) -> np.ndarray:
        """Generates the average filter kernel
        Returns:
            kernel (np.ndarray, 2D): The average filter kernel
        """
        kernel = np.ones((self.kernel_size, self.kernel_size), dtype=np.float32)
        kernel /= np.sum(kernel)
        return kernel

    def Filter(self, image: np.ndarray) -> np.ndarray:
        """Apply average filter to an image using convolution
        Args:
            image (np.ndarray, 2D): The input image to be filtered
        Returns:
            image (np.ndarray, 2D): The filtered image
        """
        return CVMath.Convolve(image, self.kernel)


class MedianFilter(IFilter):
    def __init__(self, kernel_size: int = 5):
        """Constructor for MedianFilter class
        Args:
            kernel_size (int, optional): Size of the median filter kernel. Defaults to 5.
        """
        self.kernel_size = kernel_size

    def Filter(self, image: np.ndarray) -> np.ndarray:
        """Apply median filter to an image
        Args:
            image (np.ndarray, 2D): The input image to be filtered
        Returns:
            image (np.ndarray, 2D): The filtered image
        """
        k = self.kernel_size
        h, w = image.shape
        output = np.zeros_like(image, dtype=np.float32)

        for i in range(h - k + 1):
            for j in range(w - k + 1):
                region = image[i : i + k, j : j + k]
                output[i + k // 2, j + k // 2] = np.median(region)

        return np.clip(output, 0, 255).astype(np.uint8)


class GaussianFilter(IFilter):
    def __init__(self, sigma: float = 1.0, kernel_size: int = 5):
        """Constructor for GaussianFilter class
        Args:
            sigma (float, optional): Standard deviation of the Gaussian kernel. Defaults to 1.0.
            kernel_size (int, optional): Size of the Gaussian kernel. Defaults to 5.
        """
        self.sigma = sigma
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
        return CVMath.Convolve(image, self.kernel)


class SobelFilter(IFilter):
    def __init__(self, axis: int = 0):
        """Constructor for SobelFilter class
        Args:
            axis (int, optional): Axis along which to apply the filter (0 for X-axis, 1 for Y-axis). Defaults to 0.
        """
        self.axis = axis
        if self.axis == 0:
            self.kernel = np.array(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32
            )
        elif self.axis == 1:
            self.kernel = np.array(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32
            )
        else:
            raise ValueError("Axis must be '0' or '1'")

    def Filter(self, image: np.ndarray) -> np.ndarray:
        """Apply Sobel filter to an image using convolution
        Args:
            image (np.ndarray, 2D): The input image to be filtered
        Returns:
            image (np.ndarray, 2D): The filtered image
        """
        return CVMath.Convolve(image, self.kernel)


class LaplacianFilter(IFilter):
    def __init__(self):
        """Constructor for LaplacianFilter class"""
        self.kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)

    def Filter(self, image: np.ndarray) -> np.ndarray:
        """Apply Laplacian filter to an image using convolution
        Args:
            image (np.ndarray, 2D): The input image to be filtered
        Returns:
            image (np.ndarray, 2D): The filtered image
        """
        return CVMath.Convolve(image, self.kernel)


class UnsharpMaskingFilter(IFilter):
    def __init__(self, sigma: float = 1.0, strength: float = 1.0):
        """Constructor for UnsharpMaskingFilter class
        Args:
            sigma (float, optional): Standard deviation of the Gaussian kernel. Defaults to 1.0.
            strength (float, optional): Strength of the sharpening. Defaults to 1.0.
        """
        self.sigma = sigma
        self.strength = strength
        self.gaussian = GaussianFilter(sigma, kernel_size=5)

    def Filter(self, image: np.ndarray) -> np.ndarray:
        """Apply Unsharp Masking (USM) filter to an image
        Args:
            image (np.ndarray, 2D): The input image to be filtered
        Returns:
            image (np.ndarray, 2D): The filtered image
        """
        blurred = self.gaussian.Filter(image)
        mask = image.astype(np.float32) - blurred.astype(np.float32)
        sharpened = image.astype(np.float32) + self.strength * mask
        return np.clip(sharpened, 0, 255).astype(np.uint8)


class HighBoostFilter(IFilter):
    def __init__(self, sigma: float = 1.0, A: float = 1.0):
        """Constructor for HighBoostFilter class
        Args:
            sigma (float, optional): Standard deviation of the Gaussian kernel. Defaults to 1.0.
            A (float, optional): Strength of the sharpening. Defaults to 1.0.
        """
        self.sigma = sigma
        self.A = A
        self.gaussian = GaussianFilter(sigma=sigma)

    def Filter(self, image: np.ndarray) -> np.ndarray:
        """Apply high-boost filtering to an image
        Args:
            image (np.ndarray, 2D): The input image to be filtered
        Returns:
            image (np.ndarray, 2D): The filtered image
        """
        blurred = self.gaussian.Filter(image)
        boosted = self.A * image.astype(np.float32) - blurred.astype(np.float32)
        return np.clip(boosted, 0, 255).astype(np.uint8)
