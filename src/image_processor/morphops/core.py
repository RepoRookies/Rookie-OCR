from .interface import IMorphOperation

import numpy as np


class Dilator(IMorphOperation):
    def __init__(self, kernel: np.ndarray):
        """
        Constructor for Dilator class
        Args:
            kernel (np.ndarray, 2D): The kernel to be used for dilation
        """
        if kernel is None:
            raise ValueError("Kernel must be provided")

        self.kernel = kernel
        self.kH, self.kW = kernel.shape

    def Morph(self, image: np.ndarray) -> np.ndarray:
        """
        Applies dilation to an image using a specified kernel
        Args:
            image (np.ndarray, 2D): The input image to be dilated
        Returns:
            image (np.ndarray, 2D): The dilated image
        """
        padH, padW = self.kH // 2, self.kW // 2
        padded = np.pad(
            image,
            ((padH, padH), (padW, padW)),
            mode="constant",
            constant_values=0,
        )
        output = np.zeros_like(image, dtype=np.uint8)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded[i : i + self.kH, j : j + self.kW]
                output[i, j] = np.max(region[self.kernel == 1])
        return output


class Eroder(IMorphOperation):
    def __init__(self, kernel: np.ndarray):
        """
        Constructor for Eroder class
        Args:
            kernel (np.ndarray, 2D): The kernel to be used for erosion morphological operation.
        """
        self.kernel = kernel
        self.kH, self.kW = kernel.shape

    def Morph(self, image: np.ndarray) -> np.ndarray:
        """
        Applies erosion to an image using a specified kernel
        Args:
            image (np.ndarray, 2D): The input image to be eroded
        Returns:
            image (np.ndarray, 2D): The eroded image
        """
        padH, padW = self.kH // 2, self.kW // 2
        padded = np.pad(
            image,
            ((padH, padH), (padW, padW)),
            mode="constant",
            constant_values=0,
        )
        output = np.zeros_like(image, dtype=np.uint8)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded[i : i + self.kH, j : j + self.kW]
                output[i, j] = np.min(region[self.kernel == 1])
        return output


class Opener(IMorphOperation):
    def __init__(self, kernel: np.ndarray):
        """
        Constructor for Opener class
        Args:
            kernel (np.ndarray, 2D): The kernel to be used for opening morphological operation.
        """
        self.eroder = Eroder(kernel)
        self.dilator = Dilator(kernel)

    def Morph(self, image: np.ndarray) -> np.ndarray:
        """
        Applies opening to an image using a specified kernel
        Args:
            image (np.ndarray, 2D): The input image to be opened
        Returns:
            image (np.ndarray, 2D): The opened image
        """
        eroded = self.eroder.Morph(image)
        opened = self.dilator.Morph(eroded)
        return opened


class Closer(IMorphOperation):
    def __init__(self, kernel: np.ndarray):
        """
        Constructor for Closer class
        Args:
            kernel (np.ndarray, 2D): The kernel to be used for closing morphological operation.
        """
        self.dilator = Dilator(kernel)
        self.eroder = Eroder(kernel)

    def Morph(self, image: np.ndarray) -> np.ndarray:
        """
        Applies closing to an image using a specified kernel
        Args:
            image (np.ndarray, 2D): The input image to be closed
        Returns:
            image (np.ndarray, 2D): The closed image
        """
        dilated = self.dilator.Morph(image)
        closed = self.eroder.Morph(dilated)
        return closed
