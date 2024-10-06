import cv2
import os

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, img_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img_thresh

# Example usage
if __name__ == "__main__":
    img_path = "../data/train/sample.jpg"
    processed_img = preprocess_image(img_path)
    cv2.imwrite("../results/processed_image.jpg", processed_img)
