from .interface import IThresholding
from src.dtypes import ThresholdingMode
from src.image_processor.filters import GaussianFilter

import numpy as np
from typing import Callable


class ThresholdHelper:
    @staticmethod
    def GetThresFunc(
        mode: ThresholdingMode, threshold: float, max_value: float
    ) -> Callable:
        """
        Returns pixel-wise thresholding function based on the thresholding modes: BINARY, BINARY_INV, TRUNC, TOZERO, TOZERO_INV
        """
        if mode == ThresholdingMode.BINARY:
            return lambda pixel: max_value if pixel > threshold else 0
        if mode == ThresholdingMode.BINARY_INV:
            return lambda pixel: 0 if pixel > threshold else max_value
        if mode == ThresholdingMode.TRUNC:
            return lambda pixel: threshold if pixel > threshold else pixel
        if mode == ThresholdingMode.TOZERO:
            return lambda pixel: 0 if pixel > threshold else pixel
        if mode == ThresholdingMode.TOZERO_INV:
            return lambda pixel: pixel if pixel > threshold else 0

        raise ValueError(f"Unsupported thresholding mode: {mode}")


class GlobalThresholding(IThresholding):
    def __init__(
        self, mode: ThresholdingMode, threshold: float = 127.0, max_value: float = 255.0
    ):
        """
        Constructor for GlobalThresholding class
        Args:
            mode (ThresholdingMode): The thresholding mode
            threshold (float, optional): The threshold value. Defaults to 127.0.
            max_value (float, optional): The maximum pixel value. Defaults to 255.
        """
        self.mode = mode
        self.threshold = threshold
        self.max_value = max_value
        self.thres_func = ThresholdHelper.GetThresFunc(mode, threshold, max_value)

    def ApplyThresholding(self, image: np.ndarray) -> np.ndarray:
        """
        Apply global thresholding to an image
        Args:
            image (np.ndarray, 2D): Input image to be thresholded
        Returns:
            image (np.ndarray, 2D): Output image after applying the global thresholding
        """
        result = np.zeros_like(image, dtype=np.float32)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                result[i, j] = self.thres_func(image[i, j])

        return np.clip(result, 0, self.max_value).astype(np.uint8)


class AdaptiveMeanThresholding(IThresholding):

    def __init__(
        self,
        mode: ThresholdingMode,
        block_size: int = 11,
        C: float = 3.0,
        max_value: float = 255.0,
    ):
        """
        Constructor for AdaptiveMeanThresholding class
        Args:
            mode (ThresholdingMode): The thresholding mode
            block_size (int, optional): The block size. Defaults to 11.
            C (float, optional): The constant value. Defaults to 3.0.
            max_value (float, optional): The maximum pixel value. Defaults to 255.
        """
        self.mode = mode
        self.max_value = max_value
        self.block_size = block_size
        self.C = C

    def ApplyThresholding(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive mean thresholding to an image
        Args:
            image (np.ndarray, 2D): Input image to be thresholded
        Returns:
            image (np.ndarray, 2D): Output image after applying the adaptive mean thresholding
        """
        pad = self.block_size // 2
        padded = np.pad(image, pad, mode="reflect")
        result = np.zeros_like(image, dtype=np.float32)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded[i : i + self.block_size, j : j + self.block_size]
                local_thresh = np.mean(region) - self.C
                thres_func = ThresholdHelper.GetThresFunc(
                    self.mode, local_thresh, self.max_value
                )
                result[i, j] = thres_func(image[i, j])

        return np.clip(result, 0, self.max_value).astype(np.uint8)


class AdaptiveGaussianThresholding(IThresholding):

    def __init__(
        self,
        mode: ThresholdingMode,
        block_size: int = 11,
        sigma: float = 1.0,
        C: float = 3.0,
        max_value: float = 255.0,
    ):
        """
        Constructor for AdaptiveGaussianThresholding class
        Args:
            mode (ThresholdingMode): The thresholding mode
            block_size (int, optional): Odd-sized neighborhood. Defaults to 11.
            sigma (float, optional): Standard deviation of the Gaussian kernel. Defaults to 1.0.
            C (float, optional): Constant subtracted from local weighted mean. Defaults to 3.0.
            max_value (float, optional): The maximum pixel value. Defaults to 255.
        """
        self.mode = mode
        self.max_value = max_value
        self.block_size = block_size
        self.sigma = sigma
        self.gaussian_kernel = GaussianFilter(
            sigma=sigma, kernel_size=self.block_size
        ).kernel
        self.C = C

    def ApplyThresholding(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive Gaussian thresholding to an image
        Args:
            image (np.ndarray, 2D): Input image to be thresholded
        Returns:
            image (np.ndarray, 2D): Output image after applying the adaptive Gaussian thresholding
        """
        pad = self.block_size // 2
        padded = np.pad(image, pad, mode="reflect")
        result = np.zeros_like(image, dtype=np.float32)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded[i : i + self.block_size, j : j + self.block_size]
                local_thresh = np.sum(region * self.gaussian_kernel) - self.C
                thres_func = ThresholdHelper.GetThresFunc(
                    self.mode, local_thresh, self.max_value
                )
                result[i, j] = thres_func(image[i, j])

        return np.clip(result, 0, self.max_value).astype(np.uint8)


class OtsuThresholding(IThresholding):
    def __init__(self, mode: ThresholdingMode, max_value: float = 255.0):
        """
        Constructor for OtsuThresholding class
        Args:
            mode (ThresholdingMode): The thresholding mode
            max_value (float, optional): The maximum pixel value. Defaults to 255.
        """
        self.mode = mode
        self.max_value = max_value

    def ApplyThresholding(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Otsu thresholding to an image by computing the histogram and finding the threshold that maximizes the between-class variance
        Args:
            image (np.ndarray, 2D): Input image to be thresholded
        Returns:
            image (np.ndarray, 2D): Output image after applying the Otsu thresholding
        """
        hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
        total = image.size
        sum_total = np.sum(np.arange(256) * hist)
        sumB, wB, max_var, threshold = 0.0, 0, 0.0, 0.0

        for t in range(256):
            wB += hist[t]
            if wB == 0:
                continue
            wF = total - wB
            if wF == 0:
                break
            sumB += t * hist[t]
            mB = sumB / wB
            mF = (sum_total - sumB) / wF
            var_between = wB * wF * (mB - mF) ** 2
            if var_between > max_var:
                max_var = var_between
                threshold = t

        threshold_func = ThresholdHelper.GetThresFunc(
            self.mode, threshold, self.max_value
        )
        result = np.zeros_like(image, dtype=np.float32)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                result[i, j] = threshold_func(image[i, j])

        return np.clip(result, 0, self.max_value).astype(np.uint8)
