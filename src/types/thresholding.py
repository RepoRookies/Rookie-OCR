from enum import Enum


class ThresholdingMode(Enum):
    BINARY = "Binary"
    BINARY_INV = "Binary Inverted"
    TRUNC = "Truncate"
    TOZERO = "To Zero"
    TOZERO_INV = "To Zero Inverted"


class ThresholdingType(Enum):
    GLOBAL = "Global Thresholding"
    ADAPTIVE_MEAN = "Adaptive Mean Thresholding"
    ADAPTIVE_GAUSSIAN = "Adaptive Gaussian Thresholding"
    OTSU = "Otsu's Method"
