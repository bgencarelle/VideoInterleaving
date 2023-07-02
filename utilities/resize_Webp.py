import os
from PIL import Image

def resize_images_in_dir(dir_path, width, height):
    for filename in os.listdir(dir_path):
        if filename.endswith(".webp"):
            image_path = os.path.join(dir_path, filename)
            with Image.open(image_path) as img:
                original_width, original_height = img.size
                print(f"The original image size is {original_width} wide x {original_height} tall")

                img = img.resize((width, height))
                resized_width, resized_height = img.size
                print(f"The resized image size is {resized_width} wide x {resized_height} tall")

                img.save(image_path)

if __name__ == "__main__":
    dir_path = input("Please enter the path to the directory: ")
    width = int(input("Please enter the desired width: "))
    height = int(input("Please enter the desired height: "))
    resize_images_in_dir(dir_path, width, height)
