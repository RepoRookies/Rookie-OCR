from src.segmentation.core import *
from src.segmentation.builder import SegmentationBuilder
from src.types.segmentation import SegmentationType 

import cv2
import matplotlib.pyplot as plt


def main():
    image = cv2.imread("assets/test_deskewed.jpg", cv2.IMREAD_GRAYSCALE)
    
    _, image = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    type = SegmentationType.HPP
    line_segmenter = SegmentationBuilder.Build(SegmentationType.HPP)
    word_segmenter = SegmentationBuilder.Build(SegmentationType.VPP, min_width=50)
    char_segmenter = SegmentationBuilder.Build(SegmentationType.CCA)
    line1 = line_segmenter.Segment(image)[0]
    word1 = word_segmenter.Segment(line1)[0]
    char1 = char_segmenter.Segment(word1)[0]

    plt.subplots(1, 4, figsize=(10, 5))
    plt.suptitle(type.value)

    plt.subplot(1, 4, 1)
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(line1, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(word1, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(char1, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
