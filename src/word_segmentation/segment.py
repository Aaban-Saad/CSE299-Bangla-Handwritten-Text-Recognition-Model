import cv2
import os

def word_segmentation(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    word_boxes = [cv2.boundingRect(c) for c in contours]
    segmented_words = [image[y:y+h, x:x+w] for (x, y, w, h) in word_boxes]
    return segmented_words, word_boxes

# Example usage
if __name__ == "__main__":
    img_path = "../results/processed_image.jpg"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    words, boxes = word_segmentation(img)
    for idx, word in enumerate(words):
        cv2.imwrite(f"../results/word_{idx}.jpg", word)
