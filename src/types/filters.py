from enum import Enum, auto


class FilterType(Enum):
    AVERAGE = auto()
    MEDIAN = auto()
    GAUSSIAN = auto()
    SOBEL = auto()
    LAPLACIAN = auto()
    UNSHARP_MASKING = auto()
    HIGH_BOOST = auto()
