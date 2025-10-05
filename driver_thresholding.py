from src.thresholding.core import *
from src.thresholding.builder import ThresholdingBuilder
from src.types.thresholding import ThresholdingMode, ThresholdingType

import cv2
import matplotlib.pyplot as plt


def main():
    image = cv2.imread("assets/test.png", cv2.IMREAD_GRAYSCALE)

    modes = [
        ThresholdingMode.BINARY,
        ThresholdingMode.BINARY_INV,
        ThresholdingMode.TRUNC,
        ThresholdingMode.TOZERO,
        ThresholdingMode.TOZERO_INV,
    ]
    type = ThresholdingType.ADAPTIVE_GAUSSIAN

    plt.subplots(1, len(modes), figsize=(10, 5))
    plt.suptitle(type.value)

    for i, mode in enumerate(modes):
        th = ThresholdingBuilder.Build(type, mode)
        op = th.ApplyThresholding(image)
        plt.subplot(1, len(modes), i + 1)
        plt.imshow(op, cmap="gray")
        plt.axis("off")
        plt.title(mode.value)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
