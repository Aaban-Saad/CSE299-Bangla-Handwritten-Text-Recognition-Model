import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D

from keras import layers
import os
from src.utils.data_generator import DataGenerator

# Model Architecture
def build_crnn_model(input_shape, num_classes):
    model = tf.keras.Sequential()
    
    # CNN layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten and pass to RNN layers
    model.add(layers.Flatten())
    
    # RNN layers
    model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
    model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
    
    # Output layer
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

# Train the model
def train_model():
    input_shape = (128, 32, 1)  # Example image size
    num_classes = 100  # Adjust this based on your dataset
    
    # Build model
    model = build_crnn_model(input_shape, num_classes)
    
    # Compile model
    model.compile(optimizer='adam', loss=tf.keras.backend.ctc_batch_cost, metrics=['accuracy'])
    
    # Data generators for training and validation
    train_gen = DataGenerator("../data/train/", batch_size=32)
    val_gen = DataGenerator("../data/validation/", batch_size=32)
    
    # Model training
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=50,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint("../models/crnn_model.h5", save_best_only=True),
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ]
    )

# Example usage
if __name__ == "__main__":
    train_model()
