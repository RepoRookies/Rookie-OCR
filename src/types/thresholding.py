from enum import Enum


class ThresholdingMode(Enum):
    BINARY = "Binary Thresholding"
    BINARY_INV = "Binary Inverted Thresholding"
    TRUNC = "Truncate Thresholding"
    TOZERO = "To Zero Thresholding"
    TOZERO_INV = "To Zero Inverted Thresholding"


class ThresholdingType(Enum):
    GLOBAL = "Global Thresholding"
    ADAPTIVE_MEAN = "Adaptive Mean Thresholding"
    ADAPTIVE_GAUSSIAN = "Adaptive Gaussian Thresholding"
    OTSU = "Otsu's Method"
