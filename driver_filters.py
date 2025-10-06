from src.image_processor.filters import *
from src.utils import Plotter

import cv2


def main():
    image = cv2.imread("assets/test.png", cv2.IMREAD_GRAYSCALE)

    type = FilterType.LAPLACIAN
    flt = FilterBuilder.Build(type)
    filtered_image = flt.Filter(image)

    Plotter.PlotImages(
        [image, filtered_image],
        title=type.value,
        subtitles=["Original Image", "Filtered Image"],
    )


if __name__ == "__main__":
    main()
