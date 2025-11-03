import cv2

from src.ocr_predicter import RookieOCR
from src.utils import Aligner
from src.image_processor.filters import FilterBuilder, FilterType
from src.image_processor.thresholding import (
    ThresholdingBuilder,
    ThresholdingType,
    ThresholdingMode,
)
from src.image_processor.segmentation import SegmentationBuilder, SegmentationType
from src.image_processor.morphops import (
    Closer,
    Opener,
)
from src.utils import MorphKernelGenerator,Plotter


class OCRPipeline:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.ocr_model = RookieOCR(model_path)

    def load_image(self, image_path: str):
        image = cv2.imread(image_path, 1)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def filter_image(self, gray_image):
        unsharp_filter = FilterBuilder.Build(FilterType.UNSHARP_MASKING, strength=1.5)
        return unsharp_filter.Filter(gray_image)

    def threshold_image(self, image):
        otsu = ThresholdingBuilder.Build(
            ThresholdingType.OTSU, ThresholdingMode.BINARY_INV, max_value=255.0
        )
        return otsu.ApplyThresholding(image)

    def align_image(self, thresh_image):
        return Aligner.DeskewTextHorizontal(thresh_image)

    def segment_lines(self, aligned_image):
        return SegmentationBuilder.Build(
            SegmentationType.HPP, threshold_ratio=0.05
        ).Segment(aligned_image)

    def segment_words(self, line_image):
        morphop_word = Closer(MorphKernelGenerator.GetSquareKernel(20))
        return SegmentationBuilder.Build(
            SegmentationType.VPP, threshold_ratio=0.12, morphop=morphop_word
        ).Segment(line_image)

    def segment_chars(self, word_image):
        morphop_char = Opener(MorphKernelGenerator.GetCrossKernel(6))
        return SegmentationBuilder.Build(
            SegmentationType.COUNTOUR, morphop=morphop_char
        ).Segment(word_image)

    def recognize_word(self, char_images):
        if not char_images:
            return ""
        return self.ocr_model.recognize_word(char_images)

    def run(self, image_path: str) -> str:
        gray = self.load_image(image_path)
        filtered = self.filter_image(gray)
        thresh = self.threshold_image(filtered)
        aligned = self.align_image(thresh)
        tokens = []
        for line in self.segment_lines(aligned):
            for word in self.segment_words(line):
                chars = self.segment_chars(word)
                Plotter.PlotImages(chars)
                token = self.recognize_word(chars)
                if token:
                    tokens.append(token)
        return " ".join(tokens).strip()



