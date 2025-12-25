import numpy as np

# Recreate the ANSI LUT one last time for the renderer
_ansi_colors = [f"\033[38;5;{i}m" for i in range(256)]
_ansi_colors[16] = "\033[38;5;235m" # Fix typical black crush
ANSI_LUT = np.array(_ansi_colors)
RESET_CODE = "\033[0m"

class AsciiSprite:
    def __init__(self, filepath):
        # Load the compressed data
        data = np.load(filepath)
        self.chars = data['chars']
        self.colors = data['colors']
        self.height, self.width = self.chars.shape

    def render(self):
        """
        Instant conversion from Data -> String.
        No resizing or math happens here.
        """
        color_strings = ANSI_LUT[self.colors]
        image_grid = np.char.add(color_strings, self.chars)
        rows = ["".join(row) for row in image_grid]
        return "\r\n".join(rows) + RESET_CODE

# Usage
# player = AsciiSprite("./assets/ascii/player_idle.npz")
# print(player.render())