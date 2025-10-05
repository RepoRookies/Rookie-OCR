from src.segmentation.interface import ISegmenter

import cv2
import numpy as np
from typing import List


class HPP_Segmentation(ISegmenter):
    def __init__(self, min_height: int = 5, margin: int = 2):
        """
        Constructor for Horizontal Projection Profile Segmentation.
        Args:
            min_height (int): The minimum height for a segment to be considered a horizontal segment.
            margin (int): The pixel margin to add around the cropped line.
        """
        self.min_height = min_height
        self.margin = margin

    def Segment(self, image: np.ndarray) -> List[np.ndarray]:
        """
            Applies the Horizontal Projection Profile (HPP) segmentation algorithm to an image.
            Args:
                image (np.ndarray, 2D): The input image to be segmented.
            Returns:
                segments (List[np.ndarray]): A list of cropped images, where each image contains one segment.
        """
        hpp = np.sum(image, axis=1)
        
        start_indices = []
        end_indices = []
        in_segment = False
        for i, val in enumerate(hpp):
            if val > 0 and not in_segment:
                start_indices.append(i)
                in_segment = True
            elif val == 0 and in_segment:
                end_indices.append(i)
                in_segment = False
        if in_segment:
            end_indices.append(len(hpp))
            
        segments = []
        for start, end in zip(start_indices, end_indices):
            if end - start > self.min_height:
                segment_crop = image[max(0, start - self.margin):min(image.shape[0], end + self.margin), :]
                segments.append(segment_crop)
                
        return segments


class VPP_Segmentation(ISegmenter):
    def __init__(self, min_width: int = 3, margin: int = 2):
        """
        Constructor for Vertical Projection Profile Segmentation.
        Args:
            min_width (int): The minimum width for a segment to be considered a vertical segment.
            margin (int): The pixel margin to add around the cropped word.
        """
        self.min_width = min_width
        self.margin = margin

    def Segment(self, image: np.ndarray) -> List[np.ndarray]:
        """
            Applies the Vertical Projection Profile (VPP) segmentation algorithm to an image.
            Args:
                image (np.ndarray, 2D): The input image to be segmented.
            Returns:
                segments (List[np.ndarray]): A list of cropped images, where each image contains one segment.
        """
        if image.size == 0:
            return []
            
        vpp = np.sum(image, axis=0)
        
        segments = []
        in_segments = False
        segment_start = 0
        
        for i, val in enumerate(vpp):
            if val > 0 and not in_segments:
                segment_start = i
                in_segments = True
            elif val == 0 and in_segments:
                if i - segment_start > self.min_width:
                    segment_crop = image[:, max(0, segment_start - self.margin):min(image.shape[1], i + self.margin)]
                    segments.append(segment_crop)
                in_segments = False
                
        if in_segments:
            if len(vpp) - segment_start > self.min_width:
                segment_crop = image[:, max(0, segment_start - self.margin):min(image.shape[1], len(vpp) + self.margin)]
                segments.append(segment_crop)
                
        return segments


class CCA_Segmentation(ISegmenter):
    def __init__(self, min_height: int = 5):
        """
        Constructor for CharacterSegmenter.
        Args:
            min_height (int): The minimum height for a component to be considered a Conne.
        """
        self.min_char_height = min_height

    def Segment(self, image: np.ndarray) -> List[np.ndarray]:
        """
            Applies the Connected Component Analysis (CCA) segmentation algorithm to an image.
            Args:
                image (np.ndarray, 2D): The input image to be segmented.
            Returns:
                segments (List[np.ndarray]): A list of cropped images, sorted left-to-right, for each segment.
        """
        if image.size == 0:
            return []

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(image, 8, cv2.CV_32S)
        
        components = []
        for i in range(1, num_labels): # Skip background label 0
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            if h > self.min_char_height:
                component_crop = image[y:y+h, x:x+w]
                components.append((x, component_crop))
                
        components.sort(key=lambda item: item[0])
        
        return [comp[1] for comp in components]