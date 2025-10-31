import cv2
import numpy as np
from .interface import IInterpolator


class Upscaler(IInterpolator):
    def __init__(self, scale_factor: float = 2.0, interpolation: int = cv2.INTER_LINEAR):
        """
        Constructor for Upscaler class.
        Args:
            scale_factor (float): Factor by which to upscale the image.
            interpolation (int): OpenCV interpolation method (default: INTER_LINEAR).
        """
        if scale_factor <= 0:
            raise ValueError("Scale factor must be positive.")
        self.scale_factor = scale_factor
        self.interpolation = interpolation

    def Interpolate(self, image: np.ndarray) -> np.ndarray:
        """
        Upscales the image by a given scale factor.
        Args:
            image (np.ndarray): Input image.
        Returns:
            np.ndarray: Upscaled image.
        """
        h, w = image.shape[:2]
        new_size = (int(w * self.scale_factor), int(h * self.scale_factor))
        return cv2.resize(image, new_size, interpolation=self.interpolation)


class Downscaler(IInterpolator):
    def __init__(self, scale_factor: float = 0.5, interpolation: int = cv2.INTER_AREA):
        """
        Constructor for Downscaler class.
        Args:
            scale_factor (float): Factor by which to downscale the image.
            interpolation (int): OpenCV interpolation method (default: INTER_AREA).
        """
        if scale_factor <= 0:
            raise ValueError("Scale factor must be positive.")
        self.scale_factor = scale_factor
        self.interpolation = interpolation

    def Interpolate(self, image: np.ndarray) -> np.ndarray:
        """
        Downscales the image by a given scale factor.
        Args:
            image (np.ndarray): Input image.
        Returns:
            np.ndarray: Downscaled image.
        """
        h, w = image.shape[:2]
        new_size = (int(w * self.scale_factor), int(h * self.scale_factor))
        return cv2.resize(image, new_size, interpolation=self.interpolation)


class Resizer(IInterpolator):
    def __init__(self, target_size: tuple[int, int], interpolation: int = cv2.INTER_LINEAR):
        """
        Constructor for Resizer class.
        Args:
            target_size (tuple[int, int]): Target size (width, height) for resizing.
            interpolation (int): OpenCV interpolation method (default: INTER_LINEAR).
        """
        if not isinstance(target_size, tuple) or len(target_size) != 2:
            raise ValueError("Target size must be a tuple (width, height).")
        self.target_size = target_size
        self.interpolation = interpolation

    def Interpolate(self, image: np.ndarray) -> np.ndarray:
        """
        Resizes the image to a target size.
        Args:
            image (np.ndarray): Input image.
        Returns:
            np.ndarray: Resized image.
        """
        return cv2.resize(image, self.target_size, interpolation=self.interpolation)
