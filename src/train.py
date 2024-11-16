# import os
# import numpy as np
# import cv2
# import tensorflow as tf
# from tensorflow.keras import layers, models
# from sklearn.model_selection import train_test_split

# def load_data(data_dir):
#     images = []
#     labels = []
#     label_map = {}
#     label_index = 0

#     # Iterate through each label folder
#     for label_folder in os.listdir(data_dir):
#         label_path = os.path.join(data_dir, label_folder)
#         print(label_path)
#         if os.path.isdir(label_path):
#             if label_folder not in label_map:
#                 label_map[label_folder] = label_index
#                 label_index += 1
            
#             # Load each image in the label folder
#             for image_name in os.listdir(label_path):
#                 image_path = os.path.join(label_path, image_name)
#                 # Read and resize the image
#                 image = cv2.imread(image_path)
#                 image = cv2.resize(image, (128, 32))  # Resize to 128x32
#                 images.append(image)
#                 labels.append(label_map[label_folder])

#     return np.array(images), np.array(labels), label_map


# def build_model(num_classes):
#     model = models.Sequential()
#     model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 128, 3)))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#     model.add(layers.Flatten())
#     model.add(layers.Dense(64, activation='relu'))
#     model.add(layers.Dense(num_classes, activation='softmax'))  # Output layer

#     model.compile(optimizer='adam',
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
#     return model

# def train_model(data_dir):
#     # Load and preprocess data
#     images, labels, label_map = load_data(data_dir)

#     # Normalize image pixel values
#     images = images.astype('float32') / 255.0

#     # Split the data into training and validation sets
#     X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

#     # Build the model
#     model = build_model(len(label_map))

#     # Train the model
#     model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, batch_size=32)

#     return model, label_map


# def train_model(data_dir):
#     # Load and preprocess data
#     images, labels, label_map = load_data(data_dir)

#     # Normalize image pixel values
#     images = images.astype('float32') / 255.0

#     # Split the data into training and validation sets
#     X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

#     # Build the model
#     model = build_model(len(label_map))

#     # Train the model
#     model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=3, batch_size=24)

#     return model, label_map


# if __name__ == "__main__":
#     data_directory = '../data/train'  # Update this path to your dataset
#     model, labels = train_model(data_directory)
#     model.save('../models/aaban.keras')
#     print("Training completed. Label mapping:", labels)



import os
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import json

# from sklearn.model_selection import train_test_split


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_dir, label_map, batch_size=32, image_size=(128, 32), shuffle=True):
        self.data_dir = data_dir
        self.label_map = label_map
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.file_paths, self.labels = self._load_file_paths_and_labels()
        self.on_epoch_end()

    def _load_file_paths_and_labels(self):
        file_paths = []
        labels = []
        for label_folder, label_index in self.label_map.items():
            label_path = os.path.join(self.data_dir, label_folder)
            print(label_path)
            if os.path.isdir(label_path):
                for image_name in os.listdir(label_path):
                    image_path = os.path.join(label_path, image_name)
                    file_paths.append(image_path)
                    labels.append(label_index)
        return file_paths, labels

    def __len__(self):
        return int(len(self.file_paths) / self.batch_size)

    def __getitem__(self, index):
        batch_paths = self.file_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
        
        images = [self._load_and_preprocess_image(file_path) for file_path in batch_paths]
        images = tf.stack(images)  # Stack images into a batch tensor
        labels = tf.convert_to_tensor(batch_labels, dtype=tf.int32)
        
        return images, labels

    def on_epoch_end(self):
        if self.shuffle:
            combined = list(zip(self.file_paths, self.labels))
            np.random.shuffle(combined)
            self.file_paths, self.labels = zip(*combined)

    def _load_and_preprocess_image(self, file_path):
        image = cv2.imread(file_path)
        image = cv2.resize(image, self.image_size)  # Resize to the specified size
        image = image / 255.0  # Normalize to [0, 1]
        return image

def load_label_map(data_dir):
    label_map = {}
    label_index = 0
    for label_folder in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, label_folder)):
            label_map[label_folder] = label_index
            label_index += 1
    return label_map

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

def train_model(data_dir, batch_size=32, epochs=4):
    # Load label map
    label_map = load_label_map(data_dir)

    # Initialize data generators
    train_generator = DataGenerator(data_dir, label_map, batch_size=batch_size, shuffle=True)
    
    # Build and train the model
    model = build_model(len(label_map))
    model.fit(train_generator, epochs=epochs)

    return model, label_map



if __name__ == "__main__":
    data_directory = '../data/train'  # Update this path to your dataset
    model, labels = train_model(data_directory)
    model.save('../models/aaban.keras')
    
    # Save labels to a JSON file
    with open('../models/labels.json', 'w') as f:
        json.dump(labels, f)
    
    print("Training completed. Label mapping saved to 'labels.json':", labels)


