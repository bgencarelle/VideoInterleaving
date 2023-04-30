import cv2
import numpy as np
import time
import csv
import os
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import mido
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

def get_aspect_ratio(image_path):
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    aspect_ratio = h/w
    return aspect_ratio


def get_image_names_from_csv(file_path):
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        png_paths = [row[1:] for row in csv_reader]

    return png_paths

def read_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    return image

def create_texture(image):
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.shape[1], image.shape[0], 0, GL_RGBA, GL_UNSIGNED_BYTE, image)
    return texture_id

def display_image(texture_id, width, height):
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 1)
    glVertex2f(0, 0)
    glTexCoord2f(1, 1)
    glVertex2f(width, 0)
    glTexCoord2f(1, 0)
    glVertex2f(width, height)
    glTexCoord2f(0, 0)
    glVertex2f(0, height)
    glEnd()


def blend_images(image1, image2):
    alpha1 = image1[:, :, 3] / 255.0
    alpha2 = image2[:, :, 3] / 255.0
    blended_alpha = alpha1 + (1 - alpha1) * alpha2

    blended_image = np.zeros_like(image1)
    for c in range(3):
        blended_image[:, :, c] = (image1[:, :, c] * alpha1 + image2[:, :, c] * alpha2 * (1 - alpha1)) / blended_alpha

    blended_image[:, :, 3] = blended_alpha * 255
    return blended_image.astype(np.uint8)

def overlay_images(image1, image2):
    blended_image = blend_images(image1, image2)
    texture_id = create_texture(blended_image)
    return texture_id
def setup_blending():
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

def display_image_with_blending(texture_id, width, height):
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 1)
    glVertex2f(0, 0)
    glTexCoord2f(1, 1)
    glVertex2f(width, 0)
    glTexCoord2f(1, 0)
    glVertex2f(width, height)
    glTexCoord2f(0, 0)
    glVertex2f(0, height)
    glEnd()

def overlay_images_fast(texture_id1, texture_id2, width, height):
    setup_blending()
    display_image_with_blending(texture_id1, width, height)
    display_image_with_blending(texture_id2, width, height)

def run_display(index, png_paths, main_folder, float_folder, display):

    pygame.init()
    width, height = display
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    fps = 20
    glMatrixMode(GL_PROJECTION)
    gluOrtho2D(0, width, 0, height)

    glEnable(GL_TEXTURE_2D)

    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        main_image_path = png_paths[index][main_folder]
        main_image = read_image(main_image_path)
        float_image_path = png_paths[index][float_folder]
        float_image = read_image(float_image_path)
        texture_id2 = create_texture(float_image)
        texture_id1 = create_texture(main_image)

        overlay_images_fast(texture_id1, texture_id2, width, height)
        #overlay_id = display_overlay(texture_float, texture_main, width, height)

        part = clock.get_fps()
        #display_image(overlay_id, width, height)
        index = (index + 1) % len(png_paths)
        if index % fps == 0:
            main_folder = (main_folder + 1) % (len(png_paths[0]) - 2)
            print(part)

        pygame.display.flip()
        clock.tick(fps)  # Target 60 FPS


def main():
    csv_source = select_csv_file()
    png_paths = get_image_names_from_csv(csv_source)
    print(len(png_paths[0]))
    aspect_ratio = get_aspect_ratio(png_paths[0][0])

    width = 810
    height = int(width * aspect_ratio)
    display = (width, height)
    start_index = 0
    main_folder = 1
    float_folder = 3
    run_display(start_index, png_paths, main_folder, float_folder, display)

if __name__ == "__main__":
    main()