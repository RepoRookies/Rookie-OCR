import tensorflow as tf
from tensorflow.keras import layers, models

class Char74KModel:
    def __init__(self, rows=18, cols=12, num_classes=62):
        self.rows = rows
        self.cols = cols
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        model = models.Sequential([
            layers.Conv2D(128, (3,3), input_shape=(self.rows, self.cols, 1), activation='relu'),
            layers.Conv2D(256, (2,1), activation='relu'),
            layers.Conv2D(512, (2,1), activation='relu'),
            layers.MaxPooling2D(pool_size=(2,2)),
            layers.Flatten(),
            layers.Dense(1024, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def summary(self):
        return self.model.summary()

    def train(self, x_train, y_train, x_val, y_val, epochs=10, batch_size=128, callbacks=None):
        return self.model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )

    def save(self, path):
        self.model.save(path)



# Alternate function-based version (same architecture)

"""
def build_char74k_model(rows=18, cols=12, num_classes=62):
    model = models.Sequential([
        layers.Conv2D(128, (3,3), input_shape=(rows, cols, 1), activation='relu'),
        layers.Conv2D(256, (2,1), activation='relu'),
        layers.Conv2D(512, (2,1), activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
"""
