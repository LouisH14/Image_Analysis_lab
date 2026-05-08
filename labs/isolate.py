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


def test_alakon(int):
    return int+1

print("isolate.py loaded")

def find_card_contours(im_obj, min_area=50000, max_area=2000000):
    """
    Detects potential card contours in the image based on shape and area.
    Uses HSV extraction and morphology from lab1.py.
    """
    image_np = im_obj.get()
    
    # 1. Extract HSV channels using the lab1 utility function
    h, s, v = lab1.extract_hsv_channels(image_np)
    
    # 2. Differentiate white background. 
    # A white background typically has high Value (V) and low Saturation (S).
    # We look for regions that are either saturated (colors) or darker (shadows/ink).
    mask = (s > 0.1) | (v < 0.9)
    
    # 3. Clean up the mask using the morphology pipeline from lab1.py
    mask_morph = lab1.apply_morphology(mask.astype(np.uint8))
    
    # 4. Find contours using OpenCV
    # RETR_EXTERNAL gets the outer boundaries (the cards)
    contours, _ = cv2.findContours(mask_morph.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    card_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Filter by expected card size
        if min_area < area < max_area:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            
            # Uno cards are quadrilaterals (exactly 4 points after approximation)
            if len(approx) == 4:
                card_contours.append(approx.reshape(4, 2))
                
    return card_contours

def visualize_isolation(im_obj):
    """
    Helper function to visualize the detected cards.
    """
    contours = find_card_contours(im_obj)
    img_draw = im_obj.get()
    
    if not contours:
        print("No cards detected")
    else:
        for cnt in contours:
            cv2.drawContours(img_draw, [cnt.astype(np.int32)], -1, (0, 255, 0), 15)
            
    plt.figure(figsize=(12, 8))
    plt.imshow(img_draw)
    plt.title(f"Detected {len(contours)} cards")
    plt.axis('off')
    plt.show()

def region_growing(seeds: list[tuple], img: np.ndarray, n_max: int = 2000, **kwargs):
    """
    Run region growing on input image using seed points, adapted for Uno cards 
    on a white background.

    Args
    ----
    seeds: list of tuple
        List of seed points (row, col)
    img: np.ndarray (M, N, C)
        RGB image
    n_max: int
        Number maximum of iterations (higher than lab1 to cover large cards)
    **kwargs:
        's_thresh' (float): Saturation threshold (default 0.1)
        'v_thresh' (float): Value threshold (default 0.9)

    Return
    ------
    rg: np.ndarray (M, N)
        Binary mask of the grown region
    """
    M, N, _ = img.shape
    rg = np.zeros((M, N)).astype(bool)

    # Use HSV extraction from lab1.py to differentiate cards from white table
    _, s, v = lab1.extract_hsv_channels(img)
    
    # Thresholds adapted for white background: grow into colorful or darker pixels
    s_thresh = kwargs.get('s_thresh', 0.1)
    v_thresh = kwargs.get('v_thresh', 0.9)
    is_card = (s > s_thresh) | (v < v_thresh)
    
    for r, c in seeds:
        if 0 <= r < M and 0 <= c < N:
            rg[r, c] = True

    for i in range(n_max):
        rg_dilated = binary_dilation(rg)
        rg_new = rg_dilated & (rg == False)
        rg_added = rg_new & is_card
        
        if not np.any(rg_added):
            break
        rg = rg | rg_added

    return rg
