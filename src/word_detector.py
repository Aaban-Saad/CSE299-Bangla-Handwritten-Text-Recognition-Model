import cv2
import pytesseract
import numpy as np

# Set the path for Tesseract executable if needed
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Adjust this path as needed

def preprocess_image(image_path):
    """Load and preprocess the image for word detection."""
    # Load the image
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binarize the image
    _, bw = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    return img, bw  # Return both the original and binarized images

def extract_words(image):
    """Extract words from the image using pytesseract."""
    h, w = image.shape
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words = []

    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 0:  # Only consider words with positive confidence
            word = data['text'][i]
            x = data['left'][i]
            y = data['top'][i]
            width = data['width'][i]
            height = data['height'][i]
            y = h - y - height  # Adjust y for OpenCV's coordinate system

            words.append((word, (x, y, width, height)))

    return words

def draw_bounding_boxes(original_image, words):
    """Draw bounding boxes around detected words on the original image."""
    for _, (x, y, width, height) in words:
        cv2.rectangle(original_image, (x, y), (x + width, y + height), (0, 255, 0), 2)  # Green box

    # Show the image with bounding boxes
    cv2.imshow("Image with Bounding Boxes", original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect(image_path):
    """Main function to process the image and show words with bounding boxes."""
    original_image, bw_image = preprocess_image(image_path)
    words = extract_words(bw_image)
    draw_bounding_boxes(original_image, words)
    print(f"Detected words: {[word[0] for word in words]}")

# Example usage
if __name__ == "__main__":
    image_path = "../data/page2.png"  # Replace with your image path
    detect(image_path)
