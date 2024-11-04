import os
import json
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

def load_labels(label_file_path):
    """Load labels from a JSON file and invert the dictionary to map indices to labels."""
    with open(label_file_path, 'r') as file:
        labels_dict = json.load(file)
    # Invert the dictionary to map indices to labels
    inverted_labels = {value: key for key, value in labels_dict.items()}
    return inverted_labels

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

    # Load label names
    labels = load_labels('../models/labels.json')

    # Display predictions along with images
    num_images = len(X_test)
    plt.figure(figsize=(15, 5))

    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(X_test[i])  # Display the image
        label_name = labels.get(y_pred_classes[i], "Unknown")  # Get the label or "Unknown" if index is out of range
        plt.title(f'Predicted: {label_name}')  # Show the predicted label
        plt.axis('off')  # Hide the axes

    plt.tight_layout()
    plt.show()
    plt.savefig("predictions_output.png")  # Save to a file

    # Initialize the current line number
    current_line = None

    # Print predictions with labels
    for i, pred_index in enumerate(y_pred_classes):
        label_name = labels.get(pred_index, "Unknown")
        
        # Extract the line number from the image name
        line_number = image_names[i].split('word')[0]  # This assumes your naming convention is like 'lineXwordY'

        # Check if we are in a new line
        if current_line != line_number:
            if current_line is not None:  # If it's not the first line
                print()  # Print a newline character to separate lines
            current_line = line_number  # Update the current line

        # Print the image name and predicted label
        print(f"{label_name}", end=" ")

    # Optional: Ensure the last line ends with a newline if needed
    print()  # Ensure the output ends with a newline

    # Open a file to write the predictions
    with open('predictions_output.txt', 'w') as f:
        current_line = None  # Initialize the current line number

        for i, pred_index in enumerate(y_pred_classes):
            label_name = labels.get(pred_index, "Unknown")
            
            # Extract the line number from the image name
            line_number = image_names[i].split('word')[0]  # Assuming naming convention is like 'lineXwordY'

            # Check if we are in a new line
            if current_line != line_number:
                if current_line is not None:  # If it's not the first line
                    f.write("\n")  # Write a newline character to separate lines
                current_line = line_number  # Update the current line

            # Write the label to the file
            f.write(f"{label_name} ")  # Write each label followed by a space



if __name__ == "__main__":
    model_path = '../models/aaban.keras'  # Path to the saved model
    test_data_directory = '../data/cropped_words/'  # Path to the test data directory

    test_model(model_path, test_data_directory)
