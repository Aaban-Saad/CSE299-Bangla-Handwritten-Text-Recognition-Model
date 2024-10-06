import os
import cv2
import numpy as np
from tensorflow.keras.utils import Sequence
from src.data_preprocessing.preprocess import preprocess_image

class DataGenerator(Sequence):
    def __init__(self, data_dir, batch_size=32, img_size=(128, 32)):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.image_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir)]
    
    def __len__(self):
        return len(self.image_paths) // self.batch_size
    
    def __getitem__(self, idx):
        batch_paths = self.image_paths[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_images = np.array([self.preprocess_image(image_path) for image_path in batch_paths])
        batch_labels = ...  # Load corresponding labels
        return batch_images, batch_labels
    
    def preprocess_image(self, image_path):
        img = preprocess_image(image_path)
        return cv2.resize(img, self.img_size)

# Example usage
if __name__ == "__main__":
    train_gen = DataGenerator("../data/train/")
    for batch_x, batch_y in train_gen:
        print(batch_x.shape, batch_y.shape)
