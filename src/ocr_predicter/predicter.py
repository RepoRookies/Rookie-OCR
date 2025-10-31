from src.utils.converter import ConverterUtil
from src.image_processor.interpolator import *
from src.dtypes.interpolation import InterpolationOperationType 

import numpy as np
import cv2
from matplotlib import pyplot as plt


def recognize_word(model, chars, label_map):
    recognized = ""
    for idx, ch in enumerate(chars):
        # Ensure grayscale 2D numpy array
        if isinstance(ch, np.ndarray) and ch.ndim == 3:
            ch = cv2.cvtColor(ch, cv2.COLOR_BGR2GRAY)
        # Invert and resize to match model input (18×12)
        ch = ConverterUtil.ToInverted(ch)
        img = InterpolatorBuilder(InterpolationOperationType.RESIZE)
        cv2.resize(ch, (12, 18))
        img = np.expand_dims(img, axis=(0, -1))  # shape (1,18,12,1)
        img = img.astype("float32") / 255.0
        # Predict
        pred = model.predict(img, verbose=0)
        pred_idx = np.argmax(pred)
        # Decode using label_map → readable char
        sample_name = label_map[pred_idx]
        decoded_char = decode_sample_label(sample_name)
        recognized += decoded_char
        print(f"Char {idx}: Pred={sample_name} → '{decoded_char}'")
        # Optional visualization
        plt.imshow(ch, cmap='gray')
        plt.title(f"Predicted: {decoded_char}")
        plt.axis('off')
        plt.show()
    return recognize