import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import time
from concurrent.futures import ThreadPoolExecutor

# Function to create a small background image
def create_background_image(width=128, height=32, color=(255, 255, 255)):
    return Image.new('RGB', (width, height), color)

# Function to add random noise to an image
def add_noise(image, noise_factor=1):
    arr = np.array(image)
    noise = np.random.randint(-noise_factor, noise_factor, arr.shape, dtype='int16')
    noisy_arr = np.clip(arr + noise, 0, 255).astype('uint8')
    return Image.fromarray(noisy_arr)

from PIL import Image, ImageDraw, ImageFont
import random
import numpy as np

from PIL import Image, ImageDraw, ImageFont
import random
import numpy as np

def create_text_image(text, font_path="../custom font/AabanLipi.ttf", font_size=40, text_color=(0, 0, 0), padding=(5, 50)):
    # Load font
    font = ImageFont.truetype(font_path, font_size)
    
    # Dummy image to calculate text bounding box
    dummy_image = Image.new('RGBA', (1, 1), (255, 255, 255, 0))
    draw = ImageDraw.Draw(dummy_image)
    
    # Get the bounding box of the text
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Add padding around the text
    image_width = text_width + 2 * padding[0]
    image_height = text_height + 2 * padding[1]
    
    # Create a new image with the calculated size and a white background
    image = Image.new('RGBA', (image_width, image_height), (255, 255, 255, 255))
    
    # Draw the text on the image, applying the padding
    draw = ImageDraw.Draw(image)
    draw.text((padding[0], padding[1]), text, font=font, fill=text_color)

    # Convert image to grayscale to simplify white row detection
    gray_image = image.convert("L")
    pixels = np.array(gray_image)

    # Define the threshold to detect non-white pixels
    white_threshold = 200

    # Crop white rows from the top
    top_crop = 0
    for i in range(pixels.shape[0]):
        if np.any(pixels[i] < white_threshold):
            top_crop = i
            break

    # Crop white rows from the bottom
    bottom_crop = pixels.shape[0]
    for i in range(pixels.shape[0] - 1, -1, -1):
        if np.any(pixels[i] < white_threshold):
            bottom_crop = i + 1
            break

    # Perform the cropping
    image = image.crop((0, top_crop, image.width, bottom_crop))

    # Apply a random rotation
    rotation_angle = random.uniform(-5, 5)  # Random rotation between -5 and 5 degrees
    image = image.rotate(rotation_angle, expand=True)

    # Apply random stretching
    stretch_factor_x = random.uniform(0.7, 1.2)  # Stretch between 70% and 120% of original width
    stretch_factor_y = random.uniform(0.7, 1.2)  # Stretch between 70% and 120% of original height
    new_size = (int(image.width * stretch_factor_x), int(image.height * stretch_factor_y))
    image = image.resize(new_size, Image.LANCZOS)

    return image



# Function to save words as images on a small 128x32 background
def save_words_as_images(input_string, font_path="arial.ttf", output_folder="output_images"):
    words = input_string.split()  # Split the string into words

    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, word in enumerate(words):
        # Create a 128x32 white background
        bg_image = create_background_image(128, 32, (255,255,255))

        # Create the transformed word image
        text_image = create_text_image(word, font_path=font_path, font_size=100)  # Small font size for 128x32

        # Ensure the word fits inside the 128x32 background by resizing if needed
        text_image_resized = text_image
        if text_image.width > 128 or text_image.height > 32:
            scale_factor = min(128 / text_image.width, (32) / text_image.height)
            # print(scale_factor)
            text_image_resized = text_image.resize((int(text_image.width * scale_factor), int(text_image.height * scale_factor)))

        # Generate a random position within the background
        # x_pos = random.randint(0, 128 - text_image_resized.width)
        x_pos = (128 - text_image_resized.width) // 2
        # y_pos = random.randint(0, 32 - text_image_resized.height)
        y_pos = (32 - text_image_resized.height) //2 #+ random.randint(-5, 5)

        # Paste the text image onto the background at a random position
        bg_image.paste(text_image_resized, (x_pos, y_pos), text_image_resized)

        # Save the image
        if not os.path.exists(f"{output_folder}/{word}"):
            os.makedirs(f"{output_folder}/{word}")
        timestamp = time.time()
        output_path = f"{output_folder}/{word}/{timestamp}.jpg"
        bg_image.save(output_path)

        # print(f"Saved word '{word}' as image: {output_path}")

# Example usage
input_string = input_text = open('unique_words.txt', 'r', encoding='utf-8').read()

font_paths = ["../custom font/AabanLipi.ttf",
 '../custom font/1.ttf',
  '../custom font/2.ttf',
    '../custom font/3.ttf',
      '../custom font/4.ttf',
        '../custom font/5.ttf',
          '../custom font/6.ttf',
            '../custom font/7.ttf',
              '../custom font/8.ttf',
                '../custom font/9.ttf',
                  '../custom font/10.ttf',
                    '../custom font/11.ttf',
                      '../custom font/12.ttf',
                        '../custom font/13.ttf',
                          '../custom font/14.ttf',
                            '../custom font/15.ttf',
                              '../custom font/16.ttf',
                                '../custom font/17.ttf',
                                  '../custom font/18.ttf',
                                    '../custom font/19.ttf',
                                      '../custom font/20.ttf',
                                        '../custom font/21.ttf',
                                        '../custom font/22.ttf',
                                        '../custom font/23.ttf',
                                        '../custom font/24.ttf',
                                        '../custom font/25.ttf',
                                        '../custom font/26.ttf',
                                        '../custom font/27.ttf',
                                        '../custom font/28.ttf',
                                        ]
output_folder = "../data/train"  # Specify your output folder here

for i in range(200):
    print("->")
    save_words_as_images(input_string, font_path=random.choice(font_paths), output_folder=output_folder)
    print(i)


