import numpy as np
import matplotlib.pyplot as plt
from typing import List


class PlotUtil:
    @staticmethod
    def PlotImage(image: np.ndarray, title: str = "", cmap: str = "gray"):
        """
        Plots an image using matplotlib.
        Args:
            image (np.ndarray, 2D): The image to be plotted.
            title (str): The title of the plot. Defaults to "".
            cmap (str): The color map of the plot. Defaults to "gray".
        """
        plt.figure(figsize=(5, 5))
        plt.imshow(image, cmap=cmap)
        plt.title(title)
        plt.axis("off")
        plt.show()

    def PlotImages(
        images: List[np.ndarray],
        title: str = "",
        subtitles: List[str] = [],
        cmap: str = "gray",
    ):
        """
        Plots a list of images using matplotlib.
        Args:
            images (List[np.ndarray]): A list of images to be plotted.
            title (str): The title of the plot. Defaults to "".
            subtitles (List[str]): A list of subtitles for each image. Defaults to [].
            cmap (str): The color map of the plot. Defaults to "gray".
        """
        plt.subplots(1, len(images), figsize=(10, 5))
        plt.suptitle(title)

        for i, image in enumerate(images):
            plt.subplot(1, len(images), i + 1)
            plt.imshow(image, cmap=cmap)
            plt.axis("off")

        if len(subtitles) > 0:
            for i, subtitle in enumerate(subtitles):
                plt.subplot(1, len(images), i + 1)
                plt.title(subtitle)

        plt.tight_layout()
        plt.show()
