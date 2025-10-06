from src.image_processor.morphops import *
from src.image_processor.thresholding import *
from src.utils import MorphKernelGenerator, Plotter


import cv2


def main():
    image = cv2.imread("assets/moon.png", cv2.IMREAD_GRAYSCALE)

    th = ThresholdingBuilder.Build(ThresholdingType.GLOBAL, ThresholdingMode.BINARY)
    th_img = th.ApplyThresholding(image)

    type = MorphOperationType.CLOSING
    kernel = MorphKernelGenerator.GetCrossKernel(3)

    mb = MorphOperationBuilder.Build(type=type, kernel=kernel)
    filtered_image = mb.Morph(image)

    th_filtered_image = mb.Morph(th_img)

    Plotter.PlotImages(
        [image, filtered_image, th_filtered_image],
        title=type.value,
        subtitles=["Original Image", "Direct Oped Image", "Thresholded Image"],
    )


if __name__ == "__main__":
    main()
