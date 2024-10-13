import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

def load_data(data_dir):
    images = []
    labels = []
    label_map = {}
    label_index = 0

    # Iterate through each label folder
    for label_folder in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_folder)
        if os.path.isdir(label_path):
            if label_folder not in label_map:
                label_map[label_folder] = label_index
                label_index += 1
            
            # Load each image in the label folder
            for image_name in os.listdir(label_path):
                image_path = os.path.join(label_path, image_name)
                # Read and resize the image
                image = cv2.imread(image_path)
                image = cv2.resize(image, (128, 32))  # Resize to 128x32
                images.append(image)
                labels.append(label_map[label_folder])

    return np.array(images), np.array(labels), label_map


def build_model(num_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 128, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))  # Output layer

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(data_dir):
    # Load and preprocess data
    images, labels, label_map = load_data(data_dir)

    # Normalize image pixel values
    images = images.astype('float32') / 255.0

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Build the model
    model = build_model(len(label_map))

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, batch_size=32)

    return model, label_map


def train_model(data_dir):
    # Load and preprocess data
    images, labels, label_map = load_data(data_dir)

    # Normalize image pixel values
    images = images.astype('float32') / 255.0

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Build the model
    model = build_model(len(label_map))

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32)

    return model, label_map


if __name__ == "__main__":
    data_directory = '../data/'  # Update this path to your dataset
    model, labels = train_model(data_directory)
    model.save('../models/aaban.keras')
    print("Training completed. Label mapping:", labels)
