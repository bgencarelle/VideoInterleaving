# VideoInterleaving

VideoInterleaving is a platform designed to enable artists to create video installations that can be synced to almost any source. The project uses PNG/webp sequences and is controlled by random data and MIDI. It's a quick and dirty solution for artists who want to create dynamic video installations with a high degree of control.

## Features

- **Frame Switching:** The project allows you to switch from frame to frame between variations of a video sequence, using an external or internal timecode sequencer.
- **Image Sequences:** By using image sequences, there is no decoding time and the project can switch sources from one frame to the next.
- **Transparency Support:** The project supports transparency in video sequences, even on low-end machines.

## Requirements

- Python
- A reasonably fast storage solution

## Installation

1. Install all of the project's dependencies.
2. Store your image sequences in a location of your choice.
3. Change the `setup` variable to `true` in `main.py`.

## Usage

The project has different modes of operation:

- **Standalone Mode**
- **Client Mode:** Includes MTC (Midi Time Code)
- **Hybrid Mode**

Server functionalities are also included in the project.

## Contributing

The project is still very much in development and was initially created based on specific needs. Contributions towards generalizing the project and improving the code style are welcome.

## Authors

- Ben Gencarelle
- ChatGPT

## Demos

Demos of the project in action can be found on [Ben's YouTube page](https://www.youtube.com/@bgencarelle).

