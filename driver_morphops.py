from src.morphops.core import *
from src.morphops.builder import MorphOperationBuilder
from src.types.morphops import MorphOperationType
from src.utils.kernel_util import MorphKernelUtil
from src.thresholding.builder import ThresholdingBuilder
from src.types.thresholding import ThresholdingMode, ThresholdingType
from src.utils.plot_util import PlotUtil

import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():
    image = cv2.imread("assets/moon.png", cv2.IMREAD_GRAYSCALE)

    th = ThresholdingBuilder.Build(ThresholdingType.GLOBAL, ThresholdingMode.BINARY)
    th_img = th.ApplyThresholding(image)

    type = MorphOperationType.CLOSING
    kernel = MorphKernelUtil.GetCrossKernel(3)

    mb = MorphOperationBuilder.Build(type=type, kernel=kernel)
    filtered_image = mb.Morph(image)

    th_filtered_image = mb.Morph(th_img)

    PlotUtil.PlotImages(
        [image, filtered_image, th_filtered_image],
        title=type.value,
        subtitles=["Original Image", "Direct Oped Image", "Thresholded Image"],
    )


if __name__ == "__main__":
    main()
