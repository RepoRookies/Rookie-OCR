from enum import Enum, auto


class SegmentationType(Enum):
    """
    Enumeration for different types of segmentation algorithms.
    """
    HPP = auto() # Horizontal Projection Profile
    VPP = auto() # Vertical Projection Profile
    CCA = auto() # Connected Component Analysis
    COUNTOUR = auto() # Contour-based segmentation