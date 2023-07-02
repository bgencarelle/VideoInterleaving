import os
import glob
import numpy as np
from PIL import Image
import webp

# Prompt for folders
folder_a = input("Enter the path for Folder A: ")
folder_b = input("Enter the path for Folder B: ")
folder_c = input("Enter the path for Folder C (created if not existing): ")

# Create Folder C if it doesn't exist
if not os.path.exists(folder_c):
    os.makedirs(folder_c)

# Fetch all .webp files from both folders, sorted alphabetically
files_a = sorted(glob.glob(os.path.join(folder_a, '*.webp')))
files_b = sorted(glob.glob(os.path.join(folder_b, '*.webp')))

# Ensure both folders have the same number of images
assert len(files_a) == len(files_b), "Folders don't have the same number of images"

for idx in range(len(files_a)):
    # Read images
    img_a = webp.imread(files_a[idx])
    img_b = webp.imread(files_b[idx])

    # Convert to PIL images to use PIL's split and merge functions
    img_a_pil = Image.fromarray(img_a)
    img_b_pil = Image.fromarray(img_b)

    # Split the images into individual bands
    bands_a = img_a_pil.split()
    bands_b = img_b_pil.split()

    # Use bands from image B and alpha band from image A
    bands_c = bands_b[0:3] + (bands_a[3],)

    # Merge bands back into single image
    img_c_pil = Image.merge("RGBA", bands_c)

    # Convert the PIL Image back to a numpy array
    img_c_np = np.array(img_c_pil)

    # Construct output path
    output_path = os.path.join(folder_c, f'out_{idx}.webp')

    # Write image C to disk
    webp.imwrite(output_path, img_c_np)
