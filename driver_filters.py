from src.filters.core import *
from src.filters.builder import FilterBuilder
from src.types.filters import FilterType

import cv2
import matplotlib.pyplot as plt


def main():
    image = cv2.imread("assets/test.png", cv2.IMREAD_GRAYSCALE)

    type = FilterType.LAPLACIAN
    flt = FilterBuilder.Build(type)
    filtered_image = flt.Filter(image)

    plt.subplots(1, 2, figsize=(10, 5))
    plt.suptitle(type.value)

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(filtered_image, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
