import cv2
import numpy as np


class AlignmentUtil:
    @staticmethod
    def DeskewTextHorizontal(image: np.ndarray) -> np.ndarray:
        """
            Deskews a binarized image by finding the minimum area bounding box
            around the text and rotating it to be horizontal.
            Args:
                image (np.ndarray, 2D): The input image to be deskewed.
            Returns:
                deskewed (np.ndarray, 2D): The deskewed image.
        """
        coords = np.column_stack(np.where(image > 0))
        coords = coords[:, ::-1].astype(np.float32) 

        rect = cv2.minAreaRect(coords)
        
        angle = rect[-1]
        width, height = rect[1]

        if height > width:
            angle -= 90
            
        print(f"Detected angle: {angle:.2f} degrees")

        (h, w) = image.shape
        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        return rotated