import shutil
from flask import Flask, request, jsonify
import os
import cv2
import json
import numpy as np
from sklearn.cluster import DBSCAN
from tensorflow.keras.models import load_model
from flask_cors import CORS

app = Flask(__name__)


CORS(app)  # Enable CORS for all routes


# Utility functions
def is_inside(big_box, small_box):
    big_x, big_y, big_w, big_h = big_box
    small_x, small_y, small_w, small_h = small_box
    return (big_x <= small_x and
            big_x + big_w >= small_x + small_w and
            big_y <= small_y and
            big_y + big_h >= small_y + small_h)

def overlap(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    return not (x1 > x2 + w2 or x2 > x1 + w1 or y1 > y2 + h2 or y2 > y1 + h1)

def merge_boxes(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x = min(x1, x2)
    y = min(y1, y2)
    w = max(x1 + w1, x2 + w2) - x
    h = max(y1 + h1, y2 + h2) - y
    return (x, y, w, h)

def load_labels_json(label_file_path):
    """Load labels from a JSON file with 'label:index' format."""
    with open(label_file_path, 'r') as file:
        raw_labels = json.load(file)  # Assuming it's a JSON string.
    
    # Convert 'label:index' format to 'index:label'
    label_dict = {}
    for label, index in raw_labels.items():
        label_dict[str(index)] = label  # Ensure index is a string for consistency

    return label_dict


def detect_words_mser(image_path, crop_output_dir="cropped_words"):
    # Ensure the output directory exists
    os.makedirs(crop_output_dir, exist_ok=True)

    # Load the image and convert to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get image dimensions
    image_height, image_width = gray.shape

    # Calculate min_area
    min_area = (image_width / 50) * (image_width / 50)

    # Initialize MSER detector
    mser = cv2.MSER_create()

    # Detect regions
    regions, _ = mser.detectRegions(gray)
    bboxes = [cv2.boundingRect(p.reshape(-1, 1, 2)) for p in regions]
    filtered_boxes = [(x, y, w, h) for (x, y, w, h) in bboxes if w * h > min_area]

    # Merge overlapping boxes
    merged_boxes = []
    while filtered_boxes:
        box = filtered_boxes.pop(0)
        for other_box in filtered_boxes[:]:
            if overlap(box, other_box):
                box = merge_boxes(box, other_box)
                filtered_boxes.remove(other_box)
        merged_boxes.append(box)

    # Remove boxes that are entirely inside another box
    final_boxes = []
    for big_box in merged_boxes:
        is_contained = any(is_inside(other_box, big_box) for other_box in merged_boxes if big_box != other_box)
        if not is_contained:
            final_boxes.append(big_box)

    # Cluster boxes into lines using DBSCAN
    y_centers = [(y + h // 2) for (x, y, w, h) in final_boxes]
    boxes_centers = np.array([[0, y] for y in y_centers])
    dbscan = DBSCAN(eps=20, min_samples=1)
    clusters = dbscan.fit_predict(boxes_centers)

    line_boxes = [[] for _ in range(max(clusters) + 1)]
    for idx, box in enumerate(final_boxes):
        line_boxes[clusters[idx]].append(box)
    line_boxes = sorted(line_boxes, key=lambda line: np.mean([box[1] for box in line]))

    # Save cropped word images
    cropped_filepaths = []
    for line_count, line in enumerate(line_boxes):
        line.sort(key=lambda b: b[0])
        for word_count, (x, y, w, h) in enumerate(line):
            cropped_word = image[y:y+h, x:x+w]
            canvas_width, canvas_height = 128, 32
            target_width, target_height = 125, 28
            padded_word = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

            aspect_ratio = w / h
            if aspect_ratio > 1:
                resize_width = target_width
                resize_height = int(resize_width / aspect_ratio)
                if resize_height > target_height:
                    resize_height = target_height
                    resize_width = int(resize_height * aspect_ratio)
            else:
                resize_height = target_height
                resize_width = int(resize_height * aspect_ratio)
                if resize_width > target_width:
                    resize_width = target_width
                    resize_height = int(resize_width / aspect_ratio)

            resized_word = cv2.resize(cropped_word, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
            y_offset = (canvas_height - resized_word.shape[0]) // 2
            x_offset = (canvas_width - resized_word.shape[1]) // 2
            padded_word[y_offset:y_offset+resized_word.shape[0], x_offset:x_offset+resized_word.shape[1]] = resized_word

            cropped_filename = os.path.join(crop_output_dir, f"line{line_count}word{word_count}.jpg")
            cv2.imwrite(cropped_filename, padded_word)
            cropped_filepaths.append(cropped_filename)

    return cropped_filepaths

# API routes
@app.route('/predict', methods=['POST'])
def predict():
    # Load the labels
    labels = load_labels_json('labels.json')

    # Get the input image from the request
    image_file = request.files.get('image')
    if not image_file:
        return jsonify({"error": "No image file provided"}), 400

    # Save the input image
    image_path = "input_image.jpg"
    image_file.save(image_path)

    # Detect words and crop
    cropped_images = detect_words_mser(image_path)

    # Load the model
    model = load_model('aaban.keras')

    # Prepare the cropped images for prediction
    images = []
    for image_path in cropped_images:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (128, 32))
        image = image.astype('float32') / 255.0
        images.append(image)
    images = np.array(images)

    # Make predictions
    y_pred = model.predict(images)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Convert predictions to labels
    predicted_text = ""
    current_line = None

    for i, pred_index in enumerate(y_pred_classes):
        label_name = labels[str(pred_index)]
        line_number = cropped_images[i].split('line')[1].split('word')[0]
        if current_line != line_number:
            if current_line is not None:
                predicted_text += "\n"
            current_line = line_number
        predicted_text += label_name + " "

    # Clean up cropped_words directory
    crop_output_dir = "cropped_words"
    if os.path.exists(crop_output_dir):
        shutil.rmtree(crop_output_dir)  # Delete the directory
        
    return jsonify({"predicted_text": predicted_text.strip()})

if __name__ == '__main__':
    app.run(debug=True)
