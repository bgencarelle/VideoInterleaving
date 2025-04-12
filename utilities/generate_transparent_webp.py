from PIL import Image
import os

# === PROMPT USER FOR INPUT FILE ===
reference_file = input("Enter the path to the reference image (e.g., reference.webp): ").strip()

# === VALIDATE REFERENCE FILE ===
if not os.path.isfile(reference_file):
    print(f"Error: File '{reference_file}' does not exist.")
    exit(1)

# === CONFIG ===
output_dir = "255_Transparent"
num_images = 2221
base_name = "blank"

# === LOAD REFERENCE AND GET SIZE ===
with Image.open(reference_file) as ref_img:
    width, height = ref_img.size

# === CREATE BLANK IMAGE ===
transparent_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))

# === CREATE OUTPUT DIRECTORY ===
os.makedirs(output_dir, exist_ok=True)

# === GENERATE FILES ===
for i in range(num_images):
    filename = f"{base_name}_{i:04d}.webp"
    filepath = os.path.join(output_dir, filename)
    transparent_img.save(filepath, format="WEBP", lossless=True)

print(f"✅ Generated {num_images} transparent WebP images at {width}x{height} in '{output_dir}'")
