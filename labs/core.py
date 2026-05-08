import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from enum import IntEnum

# If you use specific functions from lab_03_utils inside the class:
from utils.lab_03_utils import * 


# =============================================================================
# FAMILY 1: CORE & IMAGE HANDLING
# =============================================================================


BLACK = [0, 0, 0]
RED = [255, 0, 0]
GREEN = [0, 255, 0]
BLUE = [0, 0, 255]
HEIGHT = 2662
WIDTH = 4000

class IDX(IntEnum):
    IMAGE_ID = 0
    CENTER_CARD = 1
    ACTIVE_PLAYER = 2
    PLAYER_1_CARDS = 3
    PLAYER_2_CARDS = 4
    PLAYER_3_CARDS = 5
    PLAYER_4_CARDS = 6

def valid_nb(n):
    n = 80 if n > 80 else n
    print('colorized background') if n > 41 else print('white background')
    return n

class image:
    def __init__(self, image_number): # first image: image_number = 0 

        # Written informations
        self.image_number = image_number
        df = pd.read_csv("../data/train.csv")
        self.row = df.iloc[image_number]

        # Image display 
        folder = Path("../data/train_images")
        files = sorted(folder.glob("*.*"))
        files = [f for f in files if f.suffix.lower() in [".jpg", ".jpeg"]]
        im = Image.open(files[self.image_number])
        self.original = np.array(im.copy())

        # Drawings
        self.temp = self.original.copy()

    def show_info(self, idx):
        if idx > 6:
            return self.row
        else:
            return self.row.iloc[idx] 

    def display(self, temporary=False):       
        if temporary:
            plt.imshow(self.temp)
        else:
            plt.imshow(self.original)
        
    def get(self, temporary = False):
        if temporary:
            return self.temp.copy()
        else:
            return self.original.copy()

    def draw(self, y_start, y_stop, x_start, x_stop, color):
        self.temp[y_start:y_stop, x_start:x_stop] = color
        """border_thickness = 15 # Adjust as needed
        # Top border
        self.temp[y_start:min(y_start + border_thickness, y_stop), x_start:x_stop] = color
        # Bottom border
        self.temp[max(y_start, y_stop - border_thickness):y_stop, x_start:x_stop] = color
        # Left and Right borders (avoid overwriting corners if thickness is high)
        self.temp[y_start:y_stop, x_start:min(x_start + border_thickness, x_stop)] = color
        self.temp[y_start:y_stop, max(x_start, x_stop - border_thickness):x_stop] = color"""

    def erease_drawings(self):
        self.temp = self.original.copy()

    def segment(self, no_player):
        if no_player == 0: # Center Card
            return self.original[HEIGHT//4:HEIGHT*3//4, WIDTH//4:WIDTH*3//4].copy()
        if no_player == 1:
            return self.original[HEIGHT//2 +200:HEIGHT, 900:WIDTH-700].copy()
        if no_player == 2:
            return self.original[0:HEIGHT-500, WIDTH-1100:WIDTH].copy() # not symmetric as the object indicating the active player is always at right
        if no_player == 3:
            return self.original[0:HEIGHT//2 -200, 900:WIDTH-900].copy()
        if no_player == 4:
            return self.original[600:HEIGHT-300, 0:WIDTH//2 -600].copy()
        else:
            print('no player not valid')