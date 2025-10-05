import numpy as np


class KernelUtil:
    @staticmethod
    def GetGaussianKernel(sigma: float = 1.0, kernel_size: int = 5) -> np.ndarray:
        """
        Returns a Gaussian kernel of the specified size and standard deviation
        Args:
            sigma (float): The standard deviation of the Gaussian kernel
            kernel_size (int): The size of the Gaussian kernel
        Returns:
            kernel (np.ndarray, 2D): The Gaussian kernel
        """
        ax = np.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        kernel /= np.sum(kernel)
        return kernel


class MorphKernelUtil(KernelUtil):
    @staticmethod
    def GetSquareKernel(size: int = 3) -> np.ndarray:
        """
        Returns a square kernel of the specified size for morphological operations
        Args:
            size (int): The size of the square kernel
        Returns:
            kernel (np.ndarray, 2D): The square kernel
        """
        return np.ones((size, size), dtype=np.uint8)

    @staticmethod
    def GetCrossKernel(size: int = 3) -> np.ndarray:
        """
        Returns a cross kernel of the specified size for morphological operations
        Args:
            size (int): The size of the cross kernel
        Returns:
            kernel (np.ndarray, 2D): The cross kernel
        """
        kernel = np.zeros((size, size), dtype=np.uint8)
        center = size // 2
        kernel[:, center] = 1
        kernel[center, :] = 1
        return kernel
