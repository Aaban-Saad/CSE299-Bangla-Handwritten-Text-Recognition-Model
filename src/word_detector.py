# import cv2
# import pytesseract
# import numpy as np

# # Set the path for Tesseract executable if needed
# pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Adjust this path as needed

# def preprocess_image(image_path):
#     """Load and preprocess the image for word detection."""
#     # Load the image
#     img = cv2.imread(image_path)

#     # Convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Binarize the image
#     _, bw = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

#     return img, bw  # Return both the original and binarized images

# def extract_words(image, lang='ben'):
#     """Extract words from the image using pytesseract with specified language."""
#     h, w = image.shape
#     data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, lang=lang)
#     words = []

#     for i in range(len(data['text'])):
#         if int(data['conf'][i]) > 0:  # Only consider words with positive confidence
#             word = data['text'][i]
#             x = data['left'][i]
#             y = data['top'][i]
#             width = data['width'][i]
#             height = data['height'][i]
#             y = h - y - height  # Adjust y for OpenCV's coordinate system

#             words.append((word, (x, y, width, height)))

#     return words

# def draw_bounding_boxes(original_image, words):
#     """Draw bounding boxes around detected words on the original image."""
#     for word, (x, y, width, height) in words:
#         cv2.rectangle(original_image, (x, y), (x + width, y + height), (0, 255, 0), 2)  # Green box
#         cv2.putText(original_image, word, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

#     # Show the image with bounding boxes
#     cv2.imshow("Image with Bounding Boxes", original_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# def detect(image_path):
#     """Main function to process the image and show words with bounding boxes."""
#     original_image, bw_image = preprocess_image(image_path)
#     words = extract_words(bw_image, lang='ben')  # Set language to Bangla ('ben')
#     draw_bounding_boxes(original_image, words)
#     print(f"Detected words: {[word[0] for word in words]}")

# # Example usage
# if __name__ == "__main__":
#     image_path = "../data/detect_box/1.jpg"  # Replace with your image path
#     detect(image_path)




















# import cv2

# def detect_words_mser(image_path, output_path="output_mser.jpg"):
#     # Load image and convert to grayscale
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Initialize MSER detector
#     mser = cv2.MSER_create()

#     # Detect regions in the image
#     regions, _ = mser.detectRegions(gray)

#     # Draw bounding boxes around detected regions
#     for p in regions:
#         x, y, w, h = cv2.boundingRect(p.reshape(-1, 1, 2))
#         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

#     # Save the result
#     cv2.imwrite(output_path, image)
#     print(f"Saved image with MSER bounding boxes as {output_path}")

# # Usage
# detect_words_mser("../data/detect_box/2.png", "../data/detect_box/1-out2.png")












import cv2
import numpy as np

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

def detect_words_mser(image_path, output_path="output_mser.jpg"):
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

    # Initialize list to store merged boxes
    merged_boxes = []

    while filtered_boxes:
        box = filtered_boxes.pop(0)
        # Merge overlapping boxes with the current box
        for other_box in filtered_boxes[:]:  # Iterate over a copy to modify `filtered_boxes`
            if overlap(box, other_box):
                # Merge boxes and update the current box
                box = merge_boxes(box, other_box)
                filtered_boxes.remove(other_box)  # Remove merged box from `filtered_boxes`

        # Append the final merged box to `merged_boxes`
        merged_boxes.append(box)

    # Second pass: Remove boxes that are entirely inside a larger box
    final_boxes = []
    for big_box in merged_boxes:
        is_contained = False
        for other_box in merged_boxes:
            if big_box != other_box and is_inside(other_box, big_box):
                # If big_box is inside another box, mark it as contained
                is_contained = True
                break
        if not is_contained:
            # Only add boxes that aren't contained within another
            final_boxes.append(big_box)

    # Draw bounding boxes around detected regions
    for (x, y, w, h) in final_boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save the result
    cv2.imwrite(output_path, image)
    print(f"Saved image with merged MSER bounding boxes as {output_path}")

# Usage
detect_words_mser("../data/detect_box/1.png", "../data/detect_box/1-out.png")
















# import cv2
# import pytesseract
# import os

# # Set the path for Tesseract executable if needed
# pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Adjust this path as needed

# # Specify the language for Tesseract
# custom_config = r'--oem 3 --psm 6 -l ben'  # Use 'ben' for Bangla

# def preprocess_image(image_path):
#     """Load and preprocess the image for word detection."""
#     # Load the image
#     img = cv2.imread(image_path)

#     # Convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Binarize the image
#     _, bw = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

#     return img, bw  # Return both the original and binarized images

# def extract_words(image):
#     """Extract words from the image using pytesseract."""
#     data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=custom_config)
#     words = []
#     h, w = image.shape[:2]

#     for i in range(len(data['text'])):
#         if int(data['conf'][i]) > 0:  # Only consider words with positive confidence
#             word = data['text'][i]
#             x = data['left'][i]
#             y = data['top'][i]
#             width = data['width'][i]
#             height = data['height'][i]
#             y = h - y - height  # Adjust y for OpenCV's coordinate system

#             words.append((word, (x, y, width, height)))

#     return words

# def save_words_as_images(original_image, words):
#     """Save each detected word as an image file."""
#     # Create a directory to save the word images
#     os.makedirs('../data/extracted', exist_ok=True)

#     # Group words by their line number
#     line_groups = {}
#     for word, bbox in words:
#         # Calculate line number based on the y-coordinate
#         line_number = bbox[1] // 30  # Assuming each line is roughly 30 pixels apart
#         if line_number not in line_groups:
#             line_groups[line_number] = []
#         line_groups[line_number].append((word, bbox))

#     # Save each word image
#     for line_index, word_data in line_groups.items():
#         for word_index, (word, (x, y, width, height)) in enumerate(word_data):
#             # Crop the word from the original image
#             word_img = original_image[y:y + height, x:x + width]
#             # Construct the filename
#             filename = f'../data/extracted/line{line_index}word{word_index}.png'
#             # Save the image
#             cv2.imwrite(filename, word_img)

# def detect_and_save_words(image_path):
#     """Main function to process the image and save words as images."""
#     original_image, bw_image = preprocess_image(image_path)
#     words = extract_words(bw_image)
#     save_words_as_images(original_image, words)
#     print(f"Extracted and saved {len(words)} words.")

# # Example usage
# if __name__ == "__main__":
#     image_path = "../data/detect_box/1.jpg"  # Replace with your image path
#     detect_and_save_words(image_path)

