from src.filters.core import GaussianFilter

import cv2
import matplotlib.pyplot as plt


def main():
    image = cv2.imread("assets/test.png", cv2.IMREAD_GRAYSCALE)
    gaussian = GaussianFilter(sigma=2, kernel_size=5)
    filtered_image = gaussian.Filter(image)

    plt.subplots(1, 2, figsize=(10, 5))

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
