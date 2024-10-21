import os
import numpy as np
import cv2
import concurrent.futures
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

def load_images_from_folder(label_folder, label_index, label_map, data_dir):
    images = []
    labels = []
    label_path = os.path.join(data_dir, label_folder)

    if label_folder not in label_map:
        label_map[label_folder] = label_index
        label_index += 1

    for image_name in os.listdir(label_path):
        image_path = os.path.join(label_path, image_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (128, 32))  # Resize to 128x32
        images.append(image)
        labels.append(label_map[label_folder])

    return images, labels, label_map

def load_data(data_dir, cache_dir):
    # Check if cached data exists
    cached_images_path = os.path.join(cache_dir, 'images.npy')
    cached_labels_path = os.path.join(cache_dir, 'labels.npy')
    if os.path.exists(cached_images_path) and os.path.exists(cached_labels_path):
        print("Loading cached data...")
        images = np.load(cached_images_path)
        labels = np.load(cached_labels_path)
        label_map = np.load(os.path.join(cache_dir, 'label_map.npy'), allow_pickle=True).item()
        return images, labels, label_map

    print("Loading data using multi-threading...")
    images = []
    labels = []
    label_map = {}
    label_index = 0

    # Load images using multi-threading
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for label_folder in os.listdir(data_dir):
            if os.path.isdir(os.path.join(data_dir, label_folder)):
                futures.append(executor.submit(load_images_from_folder, label_folder, label_index, label_map, data_dir))

        for future in concurrent.futures.as_completed(futures):
            img, lbl, label_map = future.result()
            images.extend(img)
            labels.extend(lbl)

    # Convert lists to NumPy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Save the data for future use
    os.makedirs(cache_dir, exist_ok=True)
    np.save(cached_images_path, images)
    np.save(cached_labels_path, labels)
    np.save(os.path.join(cache_dir, 'label_map.npy'), label_map)

    return images, labels, label_map

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

def train_model(data_dir, cache_dir):
    # Load and preprocess data
    images, labels, label_map = load_data(data_dir, cache_dir)

    # Normalize image pixel values
    images = images.astype('float32') / 255.0

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Build the model
    model = build_model(len(label_map))

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=4, batch_size=24)

    return model, label_map

def save_to_file(file_path, content):
    # Save the content to a text file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

if __name__ == "__main__":
    data_directory = '../data/train'  # Update this path to your dataset
    cache_directory = '../data/cache'  # Directory to save/load cached data
    model, labels = train_model(data_directory, cache_directory)
    model.save('../models/aaban.keras')
    print("Training completed. Label mapping:", labels)
    save_to_file("../data/labels.txt", str(labels))

