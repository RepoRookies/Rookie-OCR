from enum import Enum


class MorphOperationType(Enum):
    DILATION = "Dilation"
    EROSION = "Erosion"
    OPENING = "Opening"
    CLOSING = "Closing"
