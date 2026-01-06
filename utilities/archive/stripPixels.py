import os
import concurrent.futures
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import numpy as np
import cv2
import sys

# Function to process the image and remove the lines
def process_image(image_path, output_folder):
    try:
        with Image.open(image_path).convert("RGBA") as img:
            data = np.array(img)
            alpha_channel = data[:, :, 3]

            # Create a mask to identify even larger chunks (preserve them)
            large_chunk_threshold = 200  # Lower threshold to identify even larger chunks (can be adjusted)
            large_chunk_mask = alpha_channel > large_chunk_threshold

            # Apply Gaussian blur to the alpha channel to smooth out abrupt transitions
            alpha_image = Image.fromarray(alpha_channel)
            blurred_alpha = alpha_image.filter(ImageFilter.GaussianBlur(radius=10))  # Further increased blur radius
            blurred_alpha_data = np.array(blurred_alpha)

            # Removing unwanted lines by identifying groups of pixels with low alpha
            threshold = 50  # Further increased transparency threshold
            mask = (blurred_alpha_data > threshold) | large_chunk_mask  # Preserve large chunks
            cleaned_alpha = np.where(mask, blurred_alpha_data, 0)

            # Apply morphological operations to further remove thin lines
            kernel = np.ones((3, 3), np.uint8)
            cleaned_alpha = cv2.morphologyEx(cleaned_alpha, cv2.MORPH_CLOSE, kernel)
            cleaned_alpha = cv2.morphologyEx(cleaned_alpha, cv2.MORPH_OPEN, kernel)

            # Purge existing fully transparent pixels to maintain original transparency
            cleaned_alpha[alpha_channel == 0] = 0

            # Update the alpha channel with cleaned data
            data[:, :, 3] = np.where(large_chunk_mask, alpha_channel, cleaned_alpha)  # Preserve original large chunks

            # Apply sharpness filter to the image
            sharp_img = Image.fromarray(data, 'RGBA').filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

            # Apply contrast enhancement
            contrast_enhancer = ImageEnhance.Contrast(sharp_img)
            contrast_img = contrast_enhancer.enhance(1.5)  # Increase contrast by a factor of 1.5

            # Apply color enhancement
            color_enhancer = ImageEnhance.Color(contrast_img)
            enhanced_img = color_enhancer.enhance(1.2)  # Increase color intensity by a factor of 1.2

            output_path = os.path.join(output_folder, os.path.basename(image_path))
            enhanced_img.save(output_path, "WEBP")
            print(f"Processed and saved: {output_path}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# Main function to ask for folder location and process all images
def main():
    # Default folder location if not specified
    default_folder = "/Users/main1/Downloads/VideoInterleavingOld/images/mapped_webp_810/11_B_face_MoreJPG_810"
    input_folder = input(f"Enter the folder location with the .webp images (default: {default_folder}): ")
    if not input_folder.strip():
        input_folder = default_folder

    # Define output folder in the same directory as the source with '_stripped' appended
    output_folder = input_folder + "_stripped"

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all .webp files in the folder
    images = [f for f in os.listdir(input_folder) if f.lower().endswith('.webp')]
    images = [os.path.join(input_folder, img) for img in images]

    # Using ThreadPoolExecutor to process images concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(lambda img: process_image(img, output_folder), images)

    print("Processing complete.")

if __name__ == "__main__":
    main()
