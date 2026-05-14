import cv2
import numpy as np
import matplotlib.pyplot as plt
import core


def isolate_cards(zone_crop, white_background=True, plot_debug=False):
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
    s = hsv[:, :, 1]

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
    
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) 
    
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
    return cv2.bitwise_and(gray, gray, mask=mask)
