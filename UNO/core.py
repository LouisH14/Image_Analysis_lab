import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from enum import IntEnum

# If you use specific functions from lab_03_utils inside the class:
#from labs.utils.lab_03_utils import * 
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'labs'))
import lab3

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

    def display(self, temporary=False, gray_inverted=False): 
        img = self.temp.copy() if temporary else self.original.copy()
        flag = None
        if gray_inverted:   
            if img.ndim == 3:
                img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
            flag = 'gray' 
        plt.imshow(img, cmap=flag)
        
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

def get_average_image(n):
    """
    Computes the average of the first n images in the dataset.
    """
    if n <= 0:
        return None
    
    # Load the first n images using the image class
    imgs = [image(i).get() for i in range(n)]
    
    # Compute mean across the stack and cast back to uint8 for proper image format
    return np.mean(imgs, axis=0).astype(np.uint8)

def drawing(img, y_start, y_stop, x_start, x_stop, color, border_thickness=15):
    im = img.copy()
    
    # Ensure the dimensions are valid
    if y_start >= y_stop or x_start >= x_stop:
        return im
    
    """y_start, y_stop = np.clip(y_start, 0, 2662)
    x_start, x_stop = np.clip(x_start, 0, 4000)"""

    # Draw the four sides
    # Top edge
    im[y_start:min(y_start + border_thickness, y_stop), x_start:x_stop] = color
    # Bottom edge
    im[max(y_start, y_stop - border_thickness):y_stop, x_start:x_stop] = color
    # Left edge
    im[y_start:y_stop, x_start:min(x_start + border_thickness, x_stop)] = color
    # Right edge
    im[y_start:y_stop, max(x_start, x_stop - border_thickness):x_stop] = color

    return im

def plot_hsv(im):
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2YUV) 

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    plt.figure(figsize=(16, 8)) 
    plt.subplot(1, 3, 1)
    plt.title("hue")
    plt.imshow(h)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("saturation")
    plt.imshow(s)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title(f"value")
    plt.imshow(v)
    plt.axis('off')
    plt.show()

def plot_yuv(im):
    yuv = cv2.cvtColor(im, cv2.COLOR_RGB2YUV) 

    y = yuv[:, :, 0]
    u = yuv[:, :, 1]
    v = yuv[:, :, 2]

    plt.figure(figsize=(16, 8)) 
    plt.subplot(1, 3, 1)
    plt.title("Luma")
    plt.imshow(y)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Chroma U")
    plt.imshow(u)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title(f"Chroma V")
    plt.imshow(v)
    plt.axis('off')
    plt.show()