from src.image_processor.filters import *
from src.image_processor.morphops import *
from src.image_processor.segmentation import *
from src.image_processor.thresholding import *
from src.utils import *

import cv2
import os
import numpy as np
from typing import Any, Dict, Self


class ImageProcessor:
    def __init__(self, image_path: str):
        """
        Constructor for ImageProcessor class.
        Args:
            image_path (str): The path to the image file.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f'Image file not found: "{image_path}"')
        self.image = ColorConverter.ToGrayscale(cv2.imread(image_path))
        self.lines = []
        self.words = []
        self.chars = []
        self.predicted = []

    def Plot(self, title: str = "", cmap: str = "gray") -> Self:
        """
        Plots the image using matplotlib.
        Args:
            title (str): The title of the plot. Defaults to "".
            cmap (str): The color map of the plot. Defaults to "gray".
        Returns:
            self (ImageProcessor): The ImageProcessor object with the plotted image for chaining.
        """
        Plotter.PlotImage(self.image, title, cmap)
        return self

    def PlotSegmentLine(
        self,
        index: int = 0,
        title: str = "",
        cmap: str = "gray",
    ) -> Self:
        """
        Plots the segmented line at the specified index.
        Args:
            index (int, optional): The index of the line to be plotted. Defaults to 0 (first line).
            title (str, optional): The title of the plot. Defaults to "".
            cmap (str, optional): The color map of the plot. Defaults to "gray".
        Returns:
            self (ImageProcessor): The ImageProcessor object with the plotted image for chaining.
        """
        if index < 0:
            raise ValueError("Index must be non-negative")

        if len(self.lines) == 0:
            raise ValueError("Line or HPP segmentation must be performed first")

        if index >= len(self.lines):
            raise IndexError("Index out of range")

        Plotter.PlotImage(self.lines[index], title, cmap)
        return self

    def PlotSegmentWord(
        self,
        index: int = 0,
        title: str = "",
        cmap: str = "gray",
    ) -> Self:
        """
        Plots the segmented word at the specified index.
        Args:
            index (int, optional): The index of the word to be plotted. Defaults to 0 (first word).
            title (str, optional): The title of the plot. Defaults to "".
            cmap (str, optional): The color map of the plot. Defaults to "gray".
        Returns:
            self (ImageProcessor): The ImageProcessor object with the plotted image for chaining.
        """
        if index < 0:
            raise ValueError("Index must be non-negative")

        if len(self.words) == 0:
            raise ValueError("Word or VPP segmentation must be performed first")

        if index >= len(self.words):
            raise IndexError("Index out of range")

        Plotter.PlotImage(self.words[index], title, cmap)
        return self

    def PlotSegmentChar(
        self,
        index: int = 0,
        title: str = "",
        cmap: str = "gray",
    ) -> Self:
        """
        Plots the segmented character at the specified index.
        Args:
            index (int, optional): The index of the character to be plotted. Defaults to 0 (first character).
            title (str, optional): The title of the plot. Defaults to "".
            cmap (str, optional): The color map of the plot. Defaults to "gray".
        Returns:
            self (ImageProcessor): The ImageProcessor object with the plotted image for chaining.
        """
        if index < 0:
            raise ValueError("Index must be non-negative")

        if len(self.chars) == 0:
            raise ValueError("Character or CCA segmentation must be performed first")

        if index >= len(self.chars):
            raise IndexError("Index out of range")

        Plotter.PlotImage(self.chars[index], title, cmap)
        return self

    def Pad(self, padding: int = 0, pad_value: int = 0) -> Self:
        """
        Pads the image with the specified padding and pad_value.
        Args:
            padding (int, optional): The size of the padding. Defaults to 0 (No padding).
            pad_value (int, optional): The value to pad the image with. Defaults to 0.
        Returns:
            self (ImageProcessor): The ImageProcessor object with the padded image for chaining.
        """
        if padding < 0:
            raise ValueError("Padding must be non-negative")

        if padding > 0:
            self.image = Padder.Pad(self.image, padding, pad_value)

        return self

    def Unpad(self, padding: int = 0) -> Self:
        """
        Unpads the image with the specified padding.
        Args:
            padding (int, optional): The size of the padding. Defaults to 0 (No padding).
        Returns:
            self (ImageProcessor): The ImageProcessor object with the unpadded image for chaining.
        """
        if padding < 0:
            raise ValueError("Padding must be non-negative")

        if padding > 0:
            self.image = Padder.Unpad(self.image, padding)

        return self

    def Align(self) -> Self:
        """
        Aligns or deskews the image.
        Returns:
            self (ImageProcessor): The ImageProcessor object with the aligned image for chaining.
        """
        self.image = Aligner.DeskewTextHorizontal(self.image)
        return self

    def Filter(self, type: FilterType, **kwargs: Dict[str, Any]) -> Self:
        """
        Applies a filter to the image.
        Args:
            type (FilterType): The type of filter to be applied.
            **kwargs: Additional keyword arguments for specific filters.
        Returns:
            self (ImageProcessor): The ImageProcessor object with the filtered image for chaining.
        """

        flt = FilterBuilder.Build(type, **kwargs)
        self.image = flt.Filter(self.image)
        return self

    def Threshold(
        self, type: ThresholdingType, mode: ThresholdingMode, **kwargs: Dict[str, Any]
    ) -> Self:
        """
        Applies a thresholding to the image.
        Args:
            type (ThresholdingType): The type of thresholding to be applied.
            mode (ThresholdingMode): The mode of thresholding to be applied.
            **kwargs: Additional keyword arguments for specific thresholding types.
        Returns:
            self (ImageProcessor): The ImageProcessor object with the thresholded image for chaining.
        """
        th = ThresholdingBuilder.Build(type, mode, **kwargs)
        self.image = th.ApplyThresholding(self.image)
        return self

    def Morph(
        self, type: MorphOperationType, kernel: np.ndarray, **kwargs: Dict[str, Any]
    ) -> Self:
        """
        Applies a morphological operation to the image.
        Args:
            type (MorphOperationType): The type of morphological operation to be applied.
            kernel (np.ndarray, 2D): The kernel to be used for the morphological operation.
            **kwargs: Additional keyword arguments for specific morphological operations.
        Returns:
            self (ImageProcessor): The ImageProcessor object with the morphologically operated image for chaining.
        """
        mb = MorphOperationBuilder.Build(type, kernel, **kwargs)
        self.image = mb.Morph(self.image)
        return self

    def SegmentIntoLines(self, **kwargs: Dict[str, Any]) -> Self:
        """
        Applies a segmentation technique to the image and stores the lines.
        Args:
            **kwargs: Additional keyword arguments for specific segmentation types.
        Returns:
            self (ImageProcessor): The ImageProcessor object with the segmented image for chaining.
        """
        self.lines = SegmentationBuilder.Build(SegmentationType.HPP, **kwargs).Segment(
            self.image
        )
        return self

    def SegmentIntoWords(self, line_index: int = 0, **kwargs: Dict[str, Any]) -> Self:
        """
        Applies a segmentation technique to the image and stores the words.
        Args:
            line_index (int, optional): The index of the line to be segmented. Defaults to 0 (first line).
            **kwargs: Additional keyword arguments for specific segmentation types.
        Returns:
            self (ImageProcessor): The ImageProcessor object with the segmented image for chaining.
        """
        if line_index < 0:
            raise ValueError("Line index must be non-negative")

        if len(self.lines) == 0:
            raise ValueError("Line or HPP segmentation must be performed first")

        if line_index >= len(self.lines):
            raise IndexError("Index out of range")

        seg = SegmentationBuilder.Build(SegmentationType.VPP, **kwargs)
        self.words = seg.Segment(self.lines[line_index])
        return self

    def SegmentIntoChars(self, word_index: int = 0, **kwargs: Dict[str, Any]) -> Self:
        """
        Applies a segmentation technique to the image and stores the characters.
        Args:
            word_index (int, optional): The index of the word to be segmented. Defaults to 0 (first word).
            **kwargs: Additional keyword arguments for specific segmentation types.
        Returns:
            self (ImageProcessor): The ImageProcessor object with the segmented image for chaining.
        """
        if word_index < 0:
            raise ValueError("Word index must be non-negative")

        if len(self.words) == 0:
            raise ValueError("Word or VPP segmentation must be performed first")

        if word_index >= len(self.words):
            raise IndexError("Index out of range")

        seg = SegmentationBuilder.Build(SegmentationType.CCA, **kwargs)
        self.chars = seg.Segment(self.words[word_index])
        return self
    
    def OCR(self) -> Self:
        for word_ind in len(self.words):
            self.SegmentintoChars(self, word_index = word_ind)
            for char in self.chars:
                # Apply Model Present In 
                pass