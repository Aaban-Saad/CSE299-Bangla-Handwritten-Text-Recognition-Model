import os
from train import train_model
from src.word_segmentation.segment import word_segmentation
from src.data_preprocessing.preprocess import preprocess_image

def run_pipeline(image_path):
    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)
    
    # Segment words
    words, boxes = word_segmentation(preprocessed_image)
    
    # Train CRNN Model
    train_model()

# Example usage
if __name__ == "__main__":
    run_pipeline("../data/test/sample_image.jpg")
