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

def order_points(pts):
    """
    Orders 4 points in the order: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Top-left has smallest sum
    rect[2] = pts[np.argmax(s)] # Bottom-right has largest sum

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Top-right has smallest difference
    rect[3] = pts[np.argmax(diff)] # Bottom-left has largest difference
    return rect

def four_point_transform(image, pts):
    """
    Applies a perspective transform to a region defined by 4 points to get a top-down view.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Standard Uno Card dimensions (approx 2:3 ratio)
    dst_width = 300
    dst_height = 450

    dst = np.array([
        [0, 0],
        [dst_width - 1, 0],
        [dst_width - 1, dst_height - 1],
        [0, dst_height - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (dst_width, dst_height))
    return warped

def find_card_contours(image_np, min_area=100000, max_area=1500000):
    """
    Detects potential card contours in the image based on shape and area.
    Incorporates thresholding and morphology concepts from lab2.py.
    """
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np

    # Preprocessing: Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Use Otsu's Thresholding (similar to the logic in lab2.py preprocess)
    # This is more robust than Canny for cards on a table
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological operations (Closing/Opening) to solidify the card shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    card_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        # Filter by area and rectangularity (f_rect logic from lab2.py)
        if min_area < area < max_area and len(approx) >= 4:
            x, y, w, h = cv2.boundingRect(approx)
            rect_ratio = area / (w * h)
            if rect_ratio > 0.7:  # A card should fill most of its bounding box
                card_contours.append(approx.reshape(4, 2))
                
    return card_contours

def extract_isolated_cards(im_obj):
    """
    Main entry point: finds all cards in an image object and returns 
    a list of rectified (warped) card images.
    """
    original_img = im_obj.get()
    contours = find_card_contours(original_img)
    
    isolated_cards = []
    for pts in contours:
        warped = four_point_transform(original_img, pts)
        isolated_cards.append(warped)
        
    return isolated_cards

def visualize_isolation(im_obj):
    """
    Helper to visualize the result of the isolation process.
    """
    cards = extract_isolated_cards(im_obj)
    if not cards:
        print("No cards detected.")
        return

    fig, axes = plt.subplots(1, len(cards), figsize=(15, 5))
    if len(cards) == 1: axes = [axes]
    for i, card in enumerate(cards):
        axes[i].imshow(card)
        axes[i].axis('off')
    plt.show()



print("isolate.py loaded")