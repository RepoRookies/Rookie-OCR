from enum import Enum, auto

class FilterType(Enum):
    GAUSSIAN = auto()
    MEDIAN = auto()
    SOBEL = auto()
    LAPLACIAN = auto()