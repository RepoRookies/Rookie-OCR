import numpy as np


class MorphKernel:
    @staticmethod
    def Square(size: int = 3) -> np.ndarray:
        """
        Returns a square kernel of the specified size for morphological operations
        Args:
            size (int): The size of the square kernel
        Returns:
            kernel (np.ndarray, 2D): The square kernel
        """
        return np.ones((size, size), dtype=np.uint8)

    @staticmethod
    def Cross(size: int = 3) -> np.ndarray:
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
