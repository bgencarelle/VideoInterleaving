import cv2
import numpy as np
import time
import csv
import os
import pygame
import pygame.time
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import threading
from queue import Queue, Empty
import mido
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

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
    return aspect_ratio, w


def get_image_names_from_csv(file_path):
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        png_paths = [row[1:] for row in csv_reader]

    return png_paths


import imageio

import cv2

def read_image(image_path):
    image_np = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGBA)
    return image_np


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


def setup_blending():
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)


def overlay_images_fast(texture_id_main, texture_id_float, width, height):

    glClear(GL_COLOR_BUFFER_BIT)

    display_image(texture_id_float, width, height)
    display_image(texture_id_main, width, height)

def load_texture(texture_id, image):
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.shape[1], image.shape[0], 0, GL_RGBA, GL_UNSIGNED_BYTE, image)


def load_images(index, png_paths, main_folder, float_folder):

    main_image_path = png_paths[index][main_folder]
    float_image_path = png_paths[index][float_folder]

    main_image = read_image(main_image_path)
    float_image = read_image(float_image_path)

    return main_image, float_image

def run_display(index, png_paths, main_folder, float_folder, display):
    width, height = display
    fps = 30
    clock = pygame.time.Clock()
    prev_time = time.time()
    old_index = fps

    # Initialize PINGPONG related variables
    PINGPONG = True
    direction = 1

    # Create initial textures
    main_image = read_image(png_paths[index][main_folder])
    float_image = read_image(png_paths[index][float_folder])
    texture_id1 = create_texture(main_image)
    texture_id2 = create_texture(float_image)

    with ThreadPoolExecutor(max_workers=2) as executor:
        # Start the first image loading future
        image_future = executor.submit(load_images, index, png_paths, main_folder, float_folder)

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            if image_future.done():
                main_image, float_image = image_future.result()
                load_texture(texture_id1, main_image)
                load_texture(texture_id2, float_image)

                if PINGPONG:
                    # Reverse direction at boundaries
                    if index == 0 or index == len(png_paths) - 1:
                        direction = -direction

                index += direction

                current_time = time.time()
                if current_time - prev_time >= 1:
                    print("index diff is:", index-old_index)
                    old_index=index
                    prev_time = current_time
                    part = clock.get_fps()
                    main_folder = (main_folder + 1) % (len(png_paths[0]) - 2)
                    print(part)

                # Start a new future for the next images
                image_future = executor.submit(load_images, index, png_paths, main_folder, float_folder)

            overlay_images_fast(texture_id1, texture_id2, width, height)

            pygame.display.flip()
            clock.tick(fps)


def display_init(display):
    pygame.init()
    width, height = display
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    glMatrixMode(GL_PROJECTION)
    gluOrtho2D(0, width, 0, height)

    glEnable(GL_TEXTURE_2D)
    setup_blending()

def display_and_run():
    csv_source = select_csv_file()
    png_paths = get_image_names_from_csv(csv_source)
    print(len(png_paths[0]))
    aspect_ratio, width = get_aspect_ratio(png_paths[0][0])

    print(width)
    height = int(width * aspect_ratio)/2
    display = (width/2, height)
    start_index = 0
    main_folder = 6
    float_folder = 0
    display_init(display)
    run_display(start_index, png_paths, main_folder, float_folder, display)

if __name__ == "__main__":
    display_and_run()