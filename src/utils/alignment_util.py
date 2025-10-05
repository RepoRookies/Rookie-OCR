import cv2
import numpy as np


class AlignmentUtil:
    @staticmethod
    def DeskewTextHorizontal(image: np.ndarray) -> np.ndarray:
        coords = np.column_stack(np.where(image > 0))
        coords = coords[:, ::-1].astype(np.float32) 

        # 2. Get the minimum area bounding rectangle
        # It returns: (center(x,y), (width, height), angle of rotation)
        rect = cv2.minAreaRect(coords)
        
        angle = rect[-1]
        width, height = rect[1]

        if height > width:
            angle -= 90
            
        print(f"Detected angle: {angle:.2f} degrees")

        # Get image center
        (h, w) = image.shape
        center = (w // 2, h // 2)

        # Compute rotation matrix to deskew (we rotate by the negative of the detected angle)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Rotate the image
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        return rotated