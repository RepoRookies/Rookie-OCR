from src.utils.converter import ConverterUtil
from src.image_processor.interpolator import *
from src.dtypes.interpolation import InterpolationOperationType 

import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

class RookieOCR:
    def __init__(self,model_path):
        self.model = load_model(model_path)
        self.label_map = sorted(
            [f"Sample{i:03d}" for i in range(1,63)]
        )

    def decode_sample_label(self,sample_name):
        idx = int(sample_name[-3:])
        if 1 <= idx <= 10:
            return str(idx - 1)
        elif 11 <= idx <= 36:
            return chr(ord('a') + (idx - 11))
        elif 37 <= idx <= 62:
            return chr(ord('A') + (idx - 37))
        else:
            return '?'

    def recognize_word(self,chars):
        recognized = ""
        for idx, ch in enumerate(chars):
            # Ensure grayscale 2D numpy array
            if isinstance(ch, np.ndarray) and ch.ndim == 3:
                ch = cv2.cvtColor(ch, cv2.COLOR_BGR2GRAY)
            # Invert and resize to match model input (18×12)
            ch = ConverterUtil.ToInverted(ch)
            img = InterpolatorBuilder.Build(InterpolationOperationType.RESIZE, target_size = (12,18)).Interpolate(ch)
            # cv2.resize(ch, (12, 18))
            img = np.expand_dims(img, axis=(0, -1))  # shape (1,18,12,1)
            img = img.astype("float32") / 255.0
            # Predict
            pred = self.model.predict(img, verbose=0)
            pred_idx = np.argmax(pred)
            # Decode using label_map → readable char
            sample_name = self.label_map[pred_idx]
            decoded_char = self.decode_sample_label(sample_name)
            recognized += decoded_char
            print(f"Char {idx}: Pred={sample_name} → '{decoded_char}'")
            # Optional visualization
            # plt.imshow(ch, cmap='gray')
            # plt.title(f"Predicted: {decoded_char}")
            # plt.axis('off')
            # plt.show()
        return recognized