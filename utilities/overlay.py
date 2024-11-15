import os
import sys
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import threading

def overlay_images(image_a_path, image_b_path, output_path_a_over_b, output_path_b_over_a):
    # Open images
    image_a = Image.open(image_a_path).convert("RGBA").resize((810, 1080))
    image_b = Image.open(image_b_path).convert("RGBA").resize((810, 1080))
    
    # Overlay A over B
    a_over_b = Image.alpha_composite(image_b, image_a)
    print(f"Generating: {output_path_a_over_b}")
    a_over_b.save(output_path_a_over_b, format="WEBP", quality=99)

    # Overlay B over A
    b_over_a = Image.alpha_composite(image_a, image_b)
    print(f"Generating: {output_path_b_over_a}")
    b_over_a.save(output_path_b_over_a, format="WEBP", quality=99)

def prompt_user(prompt_message):
    response = input(f"{prompt_message} (y/n): ").strip().lower()
    return response == 'y'

def create_output_folders(path1, path2):
    parent_folder = os.path.dirname(path1)
    output_folder_a_over_b = os.path.join(parent_folder, f"{os.path.basename(path1)}_over_{os.path.basename(path2)}")
    output_folder_b_over_a = os.path.join(parent_folder, f"{os.path.basename(path2)}_over_{os.path.basename(path1)}")

    if not os.path.exists(output_folder_a_over_b):
        os.makedirs(output_folder_a_over_b)
    if not os.path.exists(output_folder_b_over_a):
        os.makedirs(output_folder_b_over_a)

    return output_folder_a_over_b, output_folder_b_over_a

def main():
    print("This script will overlay one RGBA image on top of another.")
    
    path1 = input("Please enter the path to the first image or folder: ").strip()
    path2 = input("Please enter the path to the second image or folder: ").strip()

    if not (os.path.exists(path1) and os.path.exists(path2)):
        print("One or both paths do not exist. Please provide valid paths.")
        sys.exit(1)

    if os.path.isfile(path1) and os.path.isfile(path2):
        if prompt_user("Proceed with overlaying the two provided images?"):
            overlay_images(
                path1,
                path2,
                "output_a_over_b.webp",
                "output_b_over_a.webp"
            )
            print("Overlay complete. Files saved as output_a_over_b.webp and output_b_over_a.webp")
        else:
            print("Operation cancelled.")
    elif os.path.isdir(path1) and os.path.isdir(path2):
        output_folder_a_over_b, output_folder_b_over_a = create_output_folders(path1, path2)

        images_a = sorted([f for f in os.listdir(path1) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])
        images_b = sorted([f for f in os.listdir(path2) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])

        if len(images_a) != len(images_b):
            print("The two folders do not contain the same number of images. Cannot proceed.")
            sys.exit(1)

        if prompt_user("Proceed with overlaying all images in the two folders?"):
            def process_images(img_a, img_b):
                img_a_path = os.path.join(path1, img_a)
                img_b_path = os.path.join(path2, img_b)
                
                output_path_a_over_b = os.path.join(output_folder_a_over_b, f"a_over_b_{os.path.splitext(img_a)[0]}.webp")
                output_path_b_over_a = os.path.join(output_folder_b_over_a, f"b_over_a_{os.path.splitext(img_b)[0]}.webp")

                overlay_images(img_a_path, img_b_path, output_path_a_over_b, output_path_b_over_a)

            with ThreadPoolExecutor(max_workers=4) as executor:
                for img_a, img_b in zip(images_a, images_b):
                    executor.submit(process_images, img_a, img_b)

            print(f"Overlay complete. Files saved in {output_folder_a_over_b} and {output_folder_b_over_a}")
        else:
            print("Operation cancelled.")
    else:
        print("Both inputs must either be files or directories.")
        sys.exit(1)

if __name__ == "__main__":
    main()
