from src.morphops.core import *
from src.morphops.builder import MorphOperationBuilder
from src.types.morphops import MorphOperationType
from src.utils.kernel import MorphKernel
from src.thresholding.builder import ThresholdingBuilder
from src.types.thresholding import ThresholdingMode, ThresholdingType

import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():
    image = cv2.imread("assets/moon.png", cv2.IMREAD_GRAYSCALE)

    th = ThresholdingBuilder.Build(ThresholdingType.GLOBAL, ThresholdingMode.BINARY)
    th_img = th.ApplyThresholding(image)

    type = MorphOperationType.CLOSING
    kernel = MorphKernel.Cross(3)

    mb = MorphOperationBuilder.Build(type=type, kernel=kernel)
    filtered_image = mb.Morph(image)

    th_filtered_image = mb.Morph(th_img)

    plt.subplots(1, 3, figsize=(10, 5))
    plt.suptitle(type.value)

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(filtered_image, cmap="gray")
    plt.title("Direct Oped Image")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(th_filtered_image, cmap="gray")
    plt.title("Thresholded Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
