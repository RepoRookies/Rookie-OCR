import glob
import numpy as np
from cv2 import resize
from matplotlib.image import imread
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorflow.keras.callbacks import Callback
from model import Char74KModel

class TQDMProgress(Callback):
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.pbar = tqdm(total=self.epochs, desc='Training Progress', ncols=100)
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.pbar.set_postfix({
            'Train Acc': f"{logs.get('accuracy', 0):.3f}",
            'Val Acc': f"{logs.get('val_accuracy', 0):.3f}",
            'Train Loss': f"{logs.get('loss', 0):.3f}",
            'Val Loss': f"{logs.get('val_loss', 0):.3f}"
        })
        self.pbar.update(1)
    def on_train_end(self, logs=None):
        self.pbar.close()

ROWS, COLS = 18, 12
DATA_PATH = "../data/English/Fnt/*"

# Data loading
images, labels = [], []
paths = sorted(glob.glob(DATA_PATH))
print(f"Found {len(paths)} character folders. Loading images...")

for folder in tqdm(paths, desc="Loading Chars74K", ncols=90):
    for image_path in glob.glob(folder + '/*'):
        img = imread(image_path)
        if img.ndim == 3:
            img = np.mean(img, axis=-1)
        img = resize(img, dsize=(COLS, ROWS))
        img = np.expand_dims(img, axis=-1)
        images.append(img)
        labels.append(int(folder[-3:]) - 1)

images = np.array(images, dtype=np.float32)
labels = np.array(labels, dtype=int)
print(f"Loaded {images.shape[0]} images of shape {images.shape[1:]} across {len(set(labels))} classes.")

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
print(f"Training samples: {len(x_train)} | Test samples: {len(x_test)}")

# Model training
cnn = Char74KModel(rows=ROWS, cols=COLS)
cnn.summary()
cnn.train(x_train, y_train, x_test, y_test, epochs=10, batch_size=128, callbacks=[TQDMProgress()])
cnn.save("char74k_cnn5.h5")
print("Model saved as char74k_cnn5.h5")
