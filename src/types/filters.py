from enum import Enum


class FilterType(Enum):
    AVERAGE = "Average/Mean Filter"
    MEDIAN = "Median Filter"
    GAUSSIAN = "Gaussian Filter"
    SOBEL = "Sobel Filter"
    LAPLACIAN = "Laplacian Filter"
    UNSHARP_MASKING = "Unsharp Masking (USM) Filter"
    HIGH_BOOST = "High-boost Filter"
