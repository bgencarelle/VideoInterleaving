
from PIL import Image, ImageOps, ImageDraw, ImageChops
import csv
import pygame.display
import pygame
import sys

import pygame.display

def select_midi_input():
    available_ports = mido.get_input_names()

    # Check if there are any available ports
    if len(available_ports) == 0:
        print("No MIDI input ports available.")
        return None

    # Check if there is only one port or all ports have the same name
    if len(set(available_ports)) == 1:
        selected_port = available_ports[0]
        print("Only one input port available, defaulting to:")
        print(selected_port)
        return selected_port

    # Prompt user to select a port
    print("Please select a MIDI input port:")
    for i, port in enumerate(available_ports):
        print(f"{i + 1}: {port}")

    while True:
        try:
            selection = input("> ")
            if selection == "":
                selected_port = available_ports[0]
                print(f"Defaulting to first MIDI input port: {selected_port}")
                return selected_port

            selection = int(selection)
            if selection not in range(1, len(available_ports) + 1):
                raise ValueError

            selected_port = available_ports[selection - 1]
            print(f"Selected port: {selected_port}")
            return selected_port

        except ValueError:
            print("Invalid selection. Please enter a number corresponding to a port.")


def select_csv_file():
    csv_dir = 'generatedPngLists'
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

    if len(csv_files) == 0:
        print(f"No .csv files found in directory {csv_dir}")
        return None

    if len(csv_files) == 1:
        print("Only one .csv file found, defaulting to:")
        selected_file = os.path.join(csv_dir, csv_files[0])
        print(selected_file)
        return selected_file

    print("Please select a .csv file to use:")
    for i, f in enumerate(csv_files):
        print(f"{i + 1}: {f}")

    while True:
        try:
            selection = int(input("> "))
            if selection not in range(1, len(csv_files) + 1):
                raise ValueError
            break
        except ValueError:
            print("Invalid selection. Please enter a number corresponding to a file.")

    selected_file = os.path.join(csv_dir, csv_files[selection - 1])
    print(f"Selected file: {selected_file}")
    return selected_file


def parse_array_file(file_path):
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        png_paths = [row[1:] for row in csv_reader]

    return png_paths

def display_index(image, index):
    font = pygame.font.Font(None, 24)
    index_text = font.render(str(index), True, (0, 0, 0))
    image.blit(index_text, (image.get_width() - index_text.get_width() - 10, 10))
    return image


def generate_background(image_path):
    img = Image.open(image_path)
    width, height = img.size
    background = Image.new('RGBA', (width, height), (30, 32, 30, 255))
    return background


def apply_float_layer_state(float_image, state):
    if state == "off":
        float_image.putalpha(0)
    elif state == "color_mask":
        alpha = float_image.split()[3]
        green_color = Image.new('RGBA', float_image.size, (0, 255, 0, 255))
        draw = ImageDraw.Draw(green_color, 'RGBA')
        draw.rectangle([(0.0, 0.0), float_image.size], fill=(0, 255, 0, 255), width=0)
        green_color.putalpha(alpha)
        float_image = green_color
    return float_image



def get_image_paths_from_csv(index, csv_file='generatedPngLists/sorted_png_stream.csv'):
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        for column in reader:
            if int(column[0]) == index:
                main_image_path = column[12]
                float_image_path = column[1]
                return main_image_path, float_image_path
    return None, None


def mix_front_back_float(index, state, background=None):
    main_image_path, float_image_path = get_image_paths_from_csv(index)

    if main_image_path is None or float_image_path is None:
        print("Image paths not found in CSV file")
        return

    main_image = Image.open(main_image_path).convert('RGBA')
    float_image = Image.open(float_image_path).convert('RGBA')

    if background is None:
        background = generate_background(main_image_path)

    mixed_image = None

    if state == 0:
        float_image.putalpha(0)
        mixed_image = Image.alpha_composite(background, main_image)
    elif state == 1:
        float_image = apply_float_layer_state(float_image, 'normal')
        mixed_image = Image.alpha_composite(background, main_image)
        mixed_image = Image.alpha_composite(mixed_image, float_image)
    elif state == 2:
        float_image = apply_float_layer_state(float_image, 'color_mask')
        mixed_image = Image.alpha_composite(background, main_image)
        mixed_image = Image.alpha_composite(mixed_image, float_image)
    elif state == 3:
        float_image = apply_float_layer_state(float_image, 'color_mask')
        mixed_image = Image.alpha_composite(background, main_image)
        mixed_image = Image.alpha_composite(mixed_image, float_image)
    elif state == 4:
        float_image = apply_float_layer_state(float_image, 'normal')
        mixed_image = Image.alpha_composite(background, main_image)
        mixed_image = Image.alpha_composite(mixed_image, float_image)

    return mixed_image



def vid_loop():
    pygame.init()

    index = 0
    state = 0  # Set the initial state
    # Get the current screen size
    screen_info = pygame.display.Info()
    screen_size = (900, 1080)

    screen = pygame.display.set_mode(screen_size, pygame.SCALED | pygame.OPENGLBLIT | pygame.DOUBLEBUF | pygame.RESIZABLE)
    pygame.display.set_caption(str(state))

    fps = 60
    clock = pygame.time.Clock()

    background = None  # Set the initial background

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Exit full screen with the ESC key
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # Get the mixed image
        mixed_image = mix_front_back_float(index, state, background)
        if mixed_image is not None:
            # Convert the PIL image to a Pygame surface
            image_data = mixed_image.tobytes()
            size = mixed_image.size
            mode = mixed_image.mode
            surface = pygame.image.fromstring(image_data, size, mode)

            # Resize the image to fit the screen
            surface = pygame.transform.scale(surface, screen_size)

            # Display the image
            screen.blit(surface, (0, 0))
            pygame.display.flip()

            # Increment the index (you can implement your own logic here)
            index += 1
            if index % 300 == 0:
                state = (state + 1) % 5  # Cycle through the states

        # Cap the frame rate
        clock.tick(fps)

    pygame.quit()

def main():
    listening_channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                          15]  # For example, listen to channels 1, 2, and 3
    midi_input_port = select_midi_input()  # Let user choose MIDI input port
    input_port = mido.open_input(midi_input_port)
    csv_source = select_csv_file()
    png_paths = parse_array_file(csv_source)

if __name__ == '__main__':
    vid_loop()
