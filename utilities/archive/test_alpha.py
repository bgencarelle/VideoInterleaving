from PIL import Image
import os

def prompt_for_image_path(prompt_text):
    while True:
        path = input(prompt_text)
        if os.path.isfile(path):
            try:
                img = Image.open(path).convert("RGBA")
                return img
            except Exception as e:
                print(f"Failed to load image: {e}")
        else:
            print("File does not exist. Please enter a valid path.")

def blend_images(base_img, overlay_img):
    # Resize overlay to match base if needed
    if overlay_img.size != base_img.size:
        overlay_img = overlay_img.resize(base_img.size)

    # Blend using alpha compositing: C_out = C_src * A_src + C_dst * (1 - A_src)
    base_pixels = base_img.load()
    overlay_pixels = overlay_img.load()

    result = Image.new("RGBA", base_img.size)
    result_pixels = result.load()

    for y in range(base_img.height):
        for x in range(base_img.width):
            r_dst, g_dst, b_dst, a_dst = base_pixels[x, y]
            r_src, g_src, b_src, a_src = overlay_pixels[x, y]
            alpha_src = a_src / 255.0
            alpha_dst = a_dst / 255.0

            r = int(r_src * alpha_src + r_dst * (1 - alpha_src))
            g = int(g_src * alpha_src + g_dst * (1 - alpha_src))
            b = int(b_src * alpha_src + b_dst * (1 - alpha_src))
            a = int(a_src + a_dst * (1 - alpha_src))

            result_pixels[x, y] = (r, g, b, a)

    return result

def main():
    print("Image Blending Script (emulating OpenGL alpha blending)")

    base_img = prompt_for_image_path("Enter path to base image: ")
    overlay_img = prompt_for_image_path("Enter path to overlay image: ")

    result = blend_images(base_img, overlay_img)

    output_path = "blended_result.png"
    result.save(output_path)
    print(f"Blended image saved to: {output_path}")

if __name__ == "__main__":
    main()
