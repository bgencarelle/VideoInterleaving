import csv
import os
import time
from collections import deque
from threading import Event

import cv2
import mido
import pygame
import numpy as np
import psutil


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


def get_image_names_from_csv(file_path):
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        png_paths = [row[1:] for row in csv_reader]

    return png_paths
def apply_float_layer_state(float_image, state):
    if state == "off":
        float_image.set_alpha(0)
    elif state == "color_mask":
        green_color = pygame.Surface(float_image.get_size(), pygame.SRCALPHA)
        green_color.fill((0, 255, 0, 255))
        float_image = pygame.Surface(float_image.get_size(), pygame.SRCALPHA)
        float_image.blit(green_color, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        float_image.blit(float_image, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
    return float_image

def generate_background(image_path):
    img = pygame.image.load(image_path)
    width, height = img.get_size()
    background = pygame.Surface((width, height), pygame.SRCALPHA)
    background.fill((30,32, 30, 255))
    return background

def mix_front_back_float_pygame(index=0, state=4, background=None):
    main_image_path = background[index][11]
    float_image_path = background[index][12]

    main_image = pygame.image.load(main_image_path).convert_alpha()
    float_image = pygame.image.load(float_image_path).convert_alpha()
    background = generate_background(main_image_path)

    mixed_image = pygame.Surface(background.get_size(), pygame.SRCALPHA)

    if state == 0:
        mixed_image.blit(background, (0, 0))
        mixed_image.blit(main_image, (0, 0))
    elif state in {1, 2, 3, 4}:
        mixed_image.blit(background, (0, 0))

        if state == 2 or state == 3:
            float_image = apply_float_layer_state(float_image, 'color_mask')
        elif state == 1 or state == 4:
            float_image = apply_float_layer_state(float_image, 'normal')

        if state == 3 or state == 4:
            mixed_image.blit(float_image, (0, 0))
            mixed_image.blit(main_image, (0, 0))
        else:
            mixed_image.blit(main_image, (0, 0))
            mixed_image.blit(float_image, (0, 0))
    else:
        print(f"Invalid state: {state}")

    return mixed_image


def display_image(png_paths):
    clock = pygame.time.Clock()
    fps = 30
    index = 0

    mixed_image = mix_front_back_float_pygame(index, 1, png_paths)
    width, height = mixed_image.get_size()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Mixed Image Display")

    running = True
    prev_time = pygame.time.get_ticks()  # Add this line to get the initial time
    while running:

        mixed_image = mix_front_back_float_pygame(index, 1, png_paths)
        screen.blit(mixed_image, (0, 0))
        pygame.display.flip()

        index += 1
        if index >= len(png_paths):
            index = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break

        clock.tick(fps)
        print(clock.get_fps())

    pygame.display.quit()


def main():
    pygame.init()
    print(pygame.display.Info())
    pygame.display.set_mode((1, 1))  # Create a dummy 1x1 display, you can set this to any size you want

    listening_channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                          15]  # For example, listen to channels 1, 2, and 3
    midi_input_port = select_midi_input()  # Let user choose MIDI input port
    input_port = mido.open_input(midi_input_port)
    csv_source = select_csv_file()
    png_paths = get_image_names_from_csv(csv_source)

    display_image(png_paths)


if __name__ == "__main__":
    main()