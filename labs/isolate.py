import os
import copy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd ########

import core
import features
import presence
import template
import lab1
from skimage.morphology import binary_dilation

#from UNO import * ###
from pathlib import Path #######
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Optional, Callable
from sklearn.metrics import accuracy_score, f1_score
from sklearn.covariance import LedoitWolf
from utils.lab_03_utils import *

WHITE_BACKGROUND_MEAN = 200
TH_INTENSITY = 18
STEP = 10 # along the same line
OFFSET = 50 # a new line has an offset from the previous one

START_POINT_PLAYER0 = (1331, 500)   # Center: Starting from mid-left
START_POINT_PLAYER1 = (2600, 100)   # P1 (Bottom-Left): x=200 is near the left edge
START_POINT_PLAYER2 = (2600, 3800)  # P2 (Right): Starting from mid-right
START_POINT_PLAYER3 = (100, 3800)   # P3 (Top): Starting from top-center
START_POINT_PLAYER4 = (100, 100)   # P4 (Left): Starting from mid-left

# Centralized configuration for player scanning
# Format: player_no: (start_point, direction_id, offset_unit_vector)
# Directions: 1:Down, 2:Right, 3:Up, 4:Left
PLAYER_SCAN_CONFIGS = {
    0: (START_POINT_PLAYER0, 2, (1, 0)),   # Center: Scan Right, shift Down
    1: (START_POINT_PLAYER1, 2, (-1, 0)),  # P1: Scan Right, shift Up
    2: (START_POINT_PLAYER2, 3, (0, -1)),  # P2: Scan Up, shift Left
    3: (START_POINT_PLAYER3, 4, (1, 0)),   # P3: Scan Left, shift Down
    4: (START_POINT_PLAYER4, 1, (0, 1))    # P4: Scan Down, shift Right
}

OPPOSITE_DIR_MAP = {1: 3, 2: 4, 3: 1, 4: 2}

def get_opposite_scan_params(y, x, direction, im_shape):
    """Calculates the starting point and direction for a return scan."""
    h, w = im_shape[:2]
    opp_dir = OPPOSITE_DIR_MAP[direction]
    
    if direction == 1:   opp_start = (h - 1, x) # Was scanning Down, start at Bottom
    elif direction == 2: opp_start = (y, w - 1) # Was scanning Right, start at Right
    elif direction == 3: opp_start = (0, x)     # Was scanning Up, start at Top
    elif direction == 4: opp_start = (y, 0)     # Was scanning Left, start at Left
    else: opp_start = (y, x)
    
    return opp_start, opp_dir


def test_alakon(int):
    return int+1

print("o.py loaded")

    
def plot_pixel_profile(im_obj, heights):
    """
    Plots the RGB channel values for all pixels at specified heights (rows).
    
    Args:
        im_obj: The image object (from core.py).
        heights: An integer or a list/range of y-coordinates (row indices) to analyze.
    """
    img = im_obj.get()
    
    # Convert a single integer height to a list for iteration
    if isinstance(heights, (int, np.integer)):
        heights = [heights]

    for h in heights:
        # Ensure the height is within bounds
        if h < 0 or h >= img.shape[0]:
            print(f"Error: Height {h} is out of bounds for image of height {img.shape[0]}")
            continue

        row_data = img[h, :, :]
        
        plt.figure(figsize=(12, 2))
        plt.plot(row_data[:, 0], color='red', label='Red')
        plt.plot(row_data[:, 1], color='green', label='Green')
        plt.plot(row_data[:, 2], color='blue', label='Blue')
        plt.title(f"Pixel Color Distribution at Height {h}")
        plt.xlabel("X coordinate (Width)")
        plt.ylabel("Intensity (0-255)")
        plt.legend()
        plt.show()

    
def in_background(y, x, im_np, inverted = False):

    """
    The intensity of the white background isn't constant. It increases toward the +y.
    Here is what we can use to solve this as a linear th:

    from sympy import symbols, Eq, solve


    x, y = symbols('x y')
    eqs = [Eq(10*x + y, 200), Eq(2500*x + y, 235)] #at height 10, intensity=200, at height 2500, intensity=235
    print(solve(eqs, [x, y]))  # {x: 2, y: 1}

    """
    # Boundary check to prevent index errors
    if y < 0 or y >= im_np.shape[0] or x < 0 or x >= im_np.shape[1]:
        return False

    pixel_is_white = np.mean(im_np[y, x]) > 200 #(0.014*y + 200) #after solving the linear syst

    # It seem a little bit strange but that is in order to pass the function as argument in still_in_background()
    if pixel_is_white :
        return False if inverted else True
    else:
        return True if inverted else False 
    

def still_in_background(im_np, start_point, direction):
    """
    Scans along a line and stops when an abrupt change in intensity is detected.
    This detects the edge of a card regardless of the background color.
    """
    y, x = start_point
    # Initialize with the intensity of the starting pixel (assumed background)
    prev_intensity = np.mean(im_np[y, x])
    
    while True:
        # Current pixel intensity
        curr_intensity = np.mean(im_np[y, x])
        
        # If the intensity changes abruptly, we hit an edge
        if abs(curr_intensity - prev_intensity) > TH_INTENSITY:
            break
            
        prev_intensity = curr_intensity

        if direction == 1: # Down
            y += STEP
        elif direction == 2: # Right
            x += STEP
        elif direction == 3: # Up
            y -= STEP
        elif direction == 4: # Left
            x -= STEP
        else: # Fallback
            x += STEP
            
        # Safety boundary check
        if y < 0 or y >= im_np.shape[0] or x < 0 or x >= im_np.shape[1]:
            break
            
    return (y, x)


def check_card_along_line(img_obj, no_player):
    im = img_obj.get()
    
    if no_player not in PLAYER_SCAN_CONFIGS:
        print("Invalid player number")
        return None

    base_start, direction, offset_unit = PLAYER_SCAN_CONFIGS[no_player]
    all_hits = []
    found_any = False
    
    # Scan multiple lines until the entire "batch" of cards is covered
    for i in range(25):
        y_curr = base_start[0] + i * offset_unit[0] * OFFSET
        x_curr = base_start[1] + i * offset_unit[1] * OFFSET
        
        # Boundary check for the start of the line
        if not (0 <= y_curr < im.shape[0] and 0 <= x_curr < im.shape[1]):
            break
        
        # Scan along the line
        hit_point = still_in_background(im, (y_curr, x_curr), direction)
        
        if 0 <= hit_point[0] < im.shape[0] and 0 <= hit_point[1] < im.shape[1]:
            opp_start, opp_dir = get_opposite_scan_params(hit_point[0], hit_point[1], direction, im.shape)
            opp_hit_point = still_in_background(im, opp_start, opp_dir)
            
            all_hits.extend([hit_point, opp_hit_point])
            found_any = True
        elif found_any:
            # We previously found cards, and now we hit background again: the batch is finished.
            break
            
    if not all_hits:
        print(f"No card found for player {no_player} after checking multiple lines.")
        return None

    y_coords = [p[0] for p in all_hits]
    x_coords = [p[1] for p in all_hits]
    return (min(y_coords), max(y_coords), min(x_coords), max(x_coords))

def visualize_card_localization(img_obj, no_player):
    """
    Visualizes the scanning process: draws the paths checked in BLUE 
    and the detected hit point in GREEN.
    """
    im = img_obj.get()
    
    if no_player not in PLAYER_SCAN_CONFIGS:
        print("Invalid player number")
        return

    base_start, direction, offset_unit = PLAYER_SCAN_CONFIGS[no_player]
    img_obj.erease_drawings() # Reset the temporary drawing buffer
    found_any = False
    
    for i in range(25):
        y_curr = base_start[0] + i * offset_unit[0] * OFFSET
        x_curr = base_start[1] + i * offset_unit[1] * OFFSET
        
        if not (0 <= y_curr < im.shape[0] and 0 <= x_curr < im.shape[1]):
            break
        
        # Find the point where the background ends
        hit_y, hit_x = still_in_background(im, (y_curr, x_curr), direction)
        
        # Determine bounds for forward drawing
        y_min, y_max = min(y_curr, hit_y), max(y_curr, hit_y)
        x_min, x_max = min(x_curr, hit_x), max(x_curr, hit_x)
        
        if direction in [2, 4]:
            img_obj.draw(y_curr - 5, y_curr + 5, x_min, x_max, core.BLUE)
        else:
            img_obj.draw(y_min, y_max, x_curr - 5, x_curr + 5, core.BLUE)

        if 0 <= hit_y < im.shape[0] and 0 <= hit_x < im.shape[1]:
            found_any = True
            # Find the opposite point
            opp_start, opp_dir = get_opposite_scan_params(hit_y, hit_x, direction, im.shape)
            opp_y, opp_x = still_in_background(im, opp_start, opp_dir)

            # Draw the opposite scan line
            if direction in [2, 4]:
                img_obj.draw(y_curr - 5, y_curr + 5, min(opp_start[1], opp_x), max(opp_start[1], opp_x), core.BLUE)
            else:
                img_obj.draw(min(opp_start[0], opp_y), max(opp_start[0], opp_y), x_curr - 5, x_curr + 5, core.BLUE)

            img_obj.draw(hit_y - 20, hit_y + 20, hit_x - 20, hit_x + 20, core.GREEN)
            img_obj.draw(opp_y - 20, opp_y + 20, opp_x - 20, opp_x + 20, core.GREEN)
        elif found_any:
            break
            
    img_obj.display(temporary=True)
    plt.title(f"Localization Debug: Player {no_player}")
    plt.show()
    
