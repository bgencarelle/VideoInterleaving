import os
from PIL import Image

def convert_black_to_white(input_folder):
    # Create output folder with '_t00_white' appended
    output_folder = f"{input_folder}_t00_white"
    os.makedirs(output_folder, exist_ok=True)

    # Loop through each file in the input folder
    for filename in os.listdir(input_folder):
        # Process only .png and .webp files
        if filename.lower().endswith(('.png', '.webp')):
            # Open image
            img_path = os.path.join(input_folder, filename)
            with Image.open(img_path) as img:
                # Convert image to RGBA if it is not already
                img = img.convert('RGBA')
                pixels = img.load()

                # Get image dimensions
                width, height = img.size

                # Replace pure black (0, 0, 0, 255) with pure white (255, 255, 255, 255)
                for x in range(width):
                    for y in range(height):
                        if pixels[x, y] == (0, 0, 0, 255):  # Check for pure black
                            pixels[x, y] = (255, 255, 255, 255)  # Set to pure white

                # Save the modified image to the output folder
                output_path = os.path.join(output_folder, filename)
                img.save(output_path)
                print(f"Converted {filename} and saved to {output_folder}")

if __name__ == "__main__":
    # Prompt user for the input folder path
    input_folder = input("Please enter the path to the folder containing the images: ").strip()
    if os.path.isdir(input_folder):
        convert_black_to_white(input_folder)
    else:
        print("The provided path is not a valid directory. Please try again.")