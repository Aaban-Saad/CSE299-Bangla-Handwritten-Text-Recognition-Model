import cv2
import numpy as np
import os
from sklearn.cluster import DBSCAN

def is_inside(big_box, small_box):
    """Check if small_box is entirely inside big_box."""
    big_x, big_y, big_w, big_h = big_box
    small_x, small_y, small_w, small_h = small_box
    return (big_x <= small_x and
            big_x + big_w >= small_x + small_w and
            big_y <= small_y and
            big_y + big_h >= small_y + small_h)

def overlap(box1, box2):
    """Check if two boxes overlap."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    return not (x1 > x2 + w2 or x2 > x1 + w1 or y1 > y2 + h2 or y2 > y1 + h1)

def merge_boxes(box1, box2):
    """Merge two overlapping boxes into a single box."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x = min(x1, x2)
    y = min(y1, y2)
    w = max(x1 + w1, x2 + w2) - x
    h = max(y1 + h1, y2 + h2) - y
    return (x, y, w, h)

def detect_words_mser(image_path, output_path="output_mser.jpg", crop_output_dir="../data/cropped_words"):
    # Ensure output directory exists
    os.makedirs(crop_output_dir, exist_ok=True)

    # Load image and convert to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get image dimensions
    image_height, image_width = gray.shape

    # Calculate min_area based on the specified formula
    min_area = (image_width / 50) * (image_width / 50)

    # Initialize MSER detector
    mser = cv2.MSER_create()

    # Detect regions in the image
    regions, _ = mser.detectRegions(gray)

    # Get bounding boxes for each region
    bboxes = [cv2.boundingRect(p.reshape(-1, 1, 2)) for p in regions]
    # Filter boxes based on area
    filtered_boxes = [(x, y, w, h) for (x, y, w, h) in bboxes if w * h > min_area]

    # Merge overlapping boxes
    merged_boxes = []
    while filtered_boxes:
        box = filtered_boxes.pop(0)
        for other_box in filtered_boxes[:]:  # Iterate over a copy to modify `filtered_boxes`
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

    # Use DBSCAN clustering to group boxes into lines
    y_centers = [(y + h // 2) for (x, y, w, h) in final_boxes]
    boxes_centers = np.array([[0, y] for y in y_centers])
    dbscan = DBSCAN(eps=20, min_samples=1)
    clusters = dbscan.fit_predict(boxes_centers)

    # Group and sort boxes by lines
    line_boxes = [[] for _ in range(max(clusters) + 1)]
    for idx, box in enumerate(final_boxes):
        line_boxes[clusters[idx]].append(box)
    line_boxes = sorted(line_boxes, key=lambda line: np.mean([box[1] for box in line]))

    # Process each line and save cropped word images
    for line_count, line in enumerate(line_boxes):
        line.sort(key=lambda b: b[0])
        for word_count, (x, y, w, h) in enumerate(line):
            cropped_word = image[y:y+h, x:x+w]

            # Draw rectangle on the original image for visualization
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # # Check black pixel percentage in the top 20 pixels of the cropped word
            # top_20_pixels = cropped_word[:h//5, :]
            # # Count pixels as black if their values are less than or equal to 20
            # black_pixel_count = np.sum(np.all(top_20_pixels <= [20, 20, 20], axis=2))

            # total_pixels = top_20_pixels.shape[0] * top_20_pixels.shape[1]
            # black_pixel_percentage = (black_pixel_count / total_pixels) * 100
            # print(black_pixel_percentage, h//10)

            # # Adjust canvas size and target dimensions based on black pixel percentage
            # if black_pixel_percentage > 10:
            #     canvas_width, canvas_height = 132, 28
            #     target_width, target_height = 120, 15
            # else:
            #     canvas_width, canvas_height = 128, 32
            #     target_width, target_height = 120, 22


            canvas_width, canvas_height = 128, 32
            target_width, target_height = 128, 32
            padded_word = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

            # Calculate aspect ratio and resize while preserving it
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

            # Center the resized word on the canvas
            y_offset = (canvas_height - resized_word.shape[0]) // 2
            x_offset = (canvas_width - resized_word.shape[1]) // 2
            padded_word[y_offset:y_offset+resized_word.shape[0], x_offset:x_offset+resized_word.shape[1]] = resized_word

            # Save the word image
            cropped_filename = os.path.join(crop_output_dir, f"line{line_count}word{word_count}.jpg")
            cv2.imwrite(cropped_filename, padded_word)

    # Save the result image with bounding boxes
    cv2.imwrite(output_path, image)
    print(f"Saved image with merged MSER bounding boxes as {output_path}")
    print(f"Cropped word images saved in '{crop_output_dir}' directory.")

# Usage
detect_words_mser("../data/detect_box/test3.png", "../data/detect_box/1-out.png")
