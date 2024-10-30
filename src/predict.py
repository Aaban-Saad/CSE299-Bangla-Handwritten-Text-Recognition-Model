import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def load_test_data(test_dir):
    images = []
    image_names = []

    # Iterate through each image in the test directory
    for image_name in os.listdir(test_dir):
        image_path = os.path.join(test_dir, image_name)
        image = cv2.imread(image_path)

        if image is not None:
            image = cv2.resize(image, (128, 32))  # Resize to 128x32
            images.append(image)
            image_names.append(image_name)  # Store the image name for reference
        else:
            print(f"Warning: Could not read image '{image_path}'.")

    return np.array(images), image_names

def test_model(model_path, test_data_dir):
    # Load the trained model
    model = load_model(model_path)
    print("Model loaded from:", model_path)

    # Load and preprocess test data
    X_test, image_names = load_test_data(test_data_dir)

    print(f"Loaded {len(X_test)} test images.")

    if len(X_test) == 0:
        print("No test data loaded. Please check the test data directory.")
        return

    # Normalize image pixel values
    X_test = X_test.astype('float32') / 255.0

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Get the class with the highest probability

    # Display predictions along with images
    num_images = len(X_test)
    plt.figure(figsize=(15, 5))

    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(X_test[i])  # Display the image
        plt.title(f'Predicted: {y_pred_classes[i]}')  # Show the predicted class
        plt.axis('off')  # Hide the axes

    plt.tight_layout()
    plt.show()
    plt.savefig("predictions_output.png")  # Save to a file


    # Print labels
    with open('../models/labels.txt', 'r') as file:
        # Read the file contents
        contents = file.read()
        # Print the contents
        print(contents)

    print(y_pred_classes)

if __name__ == "__main__":
    model_path = '../models/aaban.keras'  # Path to the saved model
    test_data_directory = '../data/test/'  # Path to the test data directory

    test_model(model_path, test_data_directory)
