import cv2
import numpy as np
import matplotlib.pyplot as plt
import core

def isolate_cards(zone_crop):
    """
    Detects individual UNO cards within a zone crop based on white borders.

    Args:
        zone_crop (np.ndarray): A NumPy array representing the cropped zone
                                 where cards are expected.

    Returns:
        list: A list of NumPy arrays, where each array is a cropped image
              of an individual card. Returns an empty list if no cards are detected.
    """
    if zone_crop is None or zone_crop.size == 0:
        return []

    # 1. Convert to grayscale
    gray = cv2.cvtColor(zone_crop, cv2.COLOR_BGR2GRAY)

    # 2. Use Canny Edge Detection instead of fixed thresholding.
    # This detects gradients (transitions) rather than absolute brightness,
    # which is much better for white-on-white or shadowed surfaces.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Morphological dilation to close gaps in the card's rectangular boundary
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # 3. Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    card_crops = []
    zone_h, zone_w = zone_crop.shape[:2]
    
    # Minimum area threshold (e.g., 5% of the zone's total area, adjust as needed)
    min_card_area = (zone_h * zone_w) * 0.01 # Reduced from 0.05 to be more flexible

    for cnt in contours:
        # Defensive check: ensure the contour is not empty.
        # An empty contour would have a size of 0.
        if cnt.size == 0:
            continue
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        # 4. Filter out noise and fragments
        # - Ignore small noise. A card should be significantly larger than typical noise.
        if area < min_card_area:
            continue
            
        # - Ignore fragments touching the edge that are too thin to be a full card.
        #   This helps remove parts of cards from adjacent zones.
        is_touching_edge = x <= 1 or y <= 1 or (x + w) >= zone_w - 1 or (y + h) >= zone_h - 1
        # A fragment might be very wide but very short, or vice-versa.
        # We assume a card has a reasonable aspect ratio and size.
        # These thresholds (0.1, 0.1) are heuristic and might need fine-tuning.
        if is_touching_edge and (w < zone_w * 0.1 or h < zone_h * 0.1):
            continue

        # 5. Extract the card
        # Ensure crop coordinates are within bounds
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(zone_w, x + w), min(zone_h, y + h)
        
        if x2 > x1 and y2 > y1: # Ensure valid crop dimensions
            card_img = zone_crop[y1:y2, x1:x2]
            card_crops.append(card_img)

    return card_crops



def visualize_isolated_cards(zone_crop, zone_index=0):
    """
    Loads an image, isolates cards in a specific zone, and plots them.
    zone_index: 0 for Center, 1-4 for Players.
    """
    
    # 3. Isolate the cards
    # Note: cards is a list of arrays with different shapes
    cards = isolate_cards(zone_crop)
    
    print(f"Found {len(cards)} cards in zone {zone_index}")
    
    if not cards:
        plt.imshow(zone_crop)
        plt.title("No cards detected in this zone")
        plt.show()
        return

    # 4. Display each card in a subplot
    fig, axes = plt.subplots(1, len(cards), figsize=(4 * len(cards), 4))
    if len(cards) == 1: axes = [axes] # Handle single card case
    
    for i, card_img in enumerate(cards):
        axes[i].imshow(card_img)
        axes[i].set_title(f"Card {i+1}\n{card_img.shape[1]}x{card_img.shape[0]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def isolate_cards_new(zone_crop, white_background=True, plot_debug=False):
    """
    Detects individual UNO cards using a custom 7-step method.

    Args:
        zone_crop (np.ndarray): The input image zone.
        white_background (bool): If True, assumes a plain white background.
        plot_debug (bool): If True, displays the intermediate mask for Step 2.
    Returns:
        list: A list of cropped card images.
    """
    if zone_crop is None or zone_crop.size == 0:
        return []

    # Step 1 — Preprocessing
    # Convert to HSV: separates Hue (color) from Saturation and Value (brightness)
    hsv = cv2.cvtColor(zone_crop, cv2.COLOR_RGB2HSV)
    # Convert to Grayscale: useful for gradient and contour work
    gray = cv2.cvtColor(zone_crop, cv2.COLOR_RGB2GRAY)    

    # Step 2 — Background Subtraction
    # Goal: Produce a binary mask where white pixels represent "card" or "border".
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    if white_background:
        # Case 1: White background. Threshold on saturation.
        # The colored interior of the card is highly saturated compared to the background.
        _, mask = cv2.threshold(s, 90, 255, cv2.THRESH_BINARY) # (im, threshold, value given to pixel>threshold, thresholding type)
        # Closing (Dilation followed by Erosion): fills small holes and connects fragments.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    else:
        # Case 2: Patterned background. Threshold on high value + low saturation.
        # This captures the white border of the cards.
        mask = cv2.inRange(hsv, (0, 0, 200), (179, 60, 255))
    
    

    #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Opening (Erosion followed by Dilation): removes small isolated noise blobs.
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Step 4 — Contour Detection
    # Find external contours on the cleaned binary mask.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # We calculate properties for each contour to prepare for filtering.
    im = zone_crop.copy()
    candidates = []
    for cnt in contours:
        rect_rotate = cv2.minAreaRect(cnt) 
        box = cv2.boxPoints(rect_rotate)
        box = np.int32(box)

        w, h = rect_rotate[1]
        rect_area = rect_rotate[1][0] * rect_rotate[1][1]

        if rect_area > 75000:
            cv2.drawContours(im, [box], 0, (0, 255, 0), 2)
            candidates.append(rect_rotate)#{'contour': cnt, 'Rect': rect_rotate, 'area': rect_area, 'aspect_ratio': aspect_ratio})
        


    if plot_debug:
        plt.figure(figsize=(24, 8)) # Made the plot bigger
        plt.subplot(1, 3, 1)
        plt.title("Original Zone Crop")
        plt.imshow(zone_crop)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Cleaned Mask (Step 3)")
        plt.imshow(mask, cmap='gray')
        plt.axis('off')

        # Step 4 Visualization: Draw contours on a copy of the original crop
        # Create a blank black canvas of the same size as the zone crop
        plt.figure(figsize=(24, 8)) 
        contour_canvas = np.zeros_like(zone_crop)
        cv2.drawContours(contour_canvas, contours, -1, (0, 255, 0), 5) # Green line, increased thickness for visibility
        plt.subplot(1, 3, 1)
        plt.title(f"Step 4: Detected Contours ({len(contours)})")
        plt.imshow(contour_canvas)
        plt.axis('off')
        plt.show()

    return candidates, im


def mask_rectangles(img, rects):
    """
    Keeps only pixels inside rects
    
    Args:
        img: image entry (numpy array)
        rects: list of rectangles ((center_x, center_y), (w, h), angle)
    
    Returns:
        image masked
    """
    if img is None or not rects:
        return np.zeros_like(img) if img is not None else None

    # Create an empty black mask of the same spatial dimensions as the image
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    for rect in rects:
        # Check if it's a rotated rectangle: ((center_x, center_y), (w, h), angle)
        if isinstance(rect[0], (tuple, list, np.ndarray)) and len(rect) == 3:
            center, size, angle = rect
            w, h = size
            h = h*1.15
            w = w*1.15
            rect_aug = (center, (w, h), angle)
            box = cv2.boxPoints(rect_aug)
            box = np.int32(box)
            cv2.fillPoly(mask, [box], 255)
    
    # Bitwise AND keeps only pixels where the mask is white (255)
    return cv2.bitwise_and(img, img, mask=mask)