import numpy as np


class PadUtil:
    @staticmethod
    def Pad(image: np.ndarray, pad_size: int = 5, pad_value: int = 0) -> np.ndarray:
        """
        Pads an image with constant values to a specified size.
        Args:
            image (np.ndarray, 2D): The input image.
            pad_size (int): The size of the padding. Defaults to 5.
            pad_value (int): The value to pad the image with. Defaults to 0.
        Returns:
            padded (np.ndarray, 2D): The padded image.
        """
        pad = pad_size // 2
        padded = np.pad(
            image,
            ((pad, pad), (pad, pad)),
            mode="constant",
            constant_values=pad_value,
        )
        return padded

    @staticmethod
    def Unpad(image: np.ndarray, pad_size: int = 5) -> np.ndarray:
        """
        Unpads an image to its original size.
        Args:
            image (np.ndarray, 2D): The input image.
            pad_size (int): The size of the padding. Defaults to 5.
        Returns:
            unpadded (np.ndarray, 2D): The unpadded image.
        """
        pad = pad_size // 2
        unpadded = image[pad:-pad, pad:-pad]
        return unpadded
