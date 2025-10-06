import cv2


class ConverterUtil:
    @staticmethod
    def ToGrayscale(image: cv2.Mat) -> cv2.Mat:
        """
        Converts an image to grayscale
        Args:
            image (cv2.Mat): The input image
        Returns:
            image (cv2.Mat): The grayscale image
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def ToBinary(image: cv2.Mat) -> cv2.Mat:
        """
        Converts an image to binary
        Args:
            image (cv2.Mat): The input image
        Returns:
            image (cv2.Mat): The binary image
        """
        return cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]
