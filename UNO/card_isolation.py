import cv2
import numpy as np
import matplotlib.pyplot as plt
import core

from scipy.ndimage import gaussian_filter1d


def isolate_cards(zone_crop, white_background=True, plot_debug=False, threshold=75000):
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

    hsv = cv2.cvtColor(zone_crop, cv2.COLOR_RGB2HSV)   
    
    if white_background:
        # On white backgrounds, colored card interiors have high saturation
        s = hsv[:, :, 1]
        _, mask = cv2.threshold(s, 90, 255, cv2.THRESH_BINARY) 
    else: # the background has colorized patterns
        # On patterned backgrounds, white card borders have low saturation and high value
        mask = cv2.inRange(hsv, (0, 0, 180), (179, 60, 255))

    # Step 3: Morphological Cleanup - bridge gaps in the detected border
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Step 4: Contour Detection
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 5 & 6: Filtering and Candidate Extraction
    im = zone_crop.copy()
    candidates = []
    contours_kept = []

    for i, cnt in enumerate(contours):
        rect_rotate = cv2.minAreaRect(cnt) 
        w, h = rect_rotate[1]
        rect_area = w * h

        if rect_area > threshold:
            box = np.int32(cv2.boxPoints(rect_rotate))
            cv2.drawContours(im, [box], 0, (0, 255, 0), 2)
            candidates.append(rect_rotate)
            contours_kept.append(cnt) # to discard contours attached to cards from other players (that appear on the border of the region)

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

        plt.figure(figsize=(24, 8)) 
        contour_canvas = np.zeros_like(zone_crop)
        cv2.drawContours(contour_canvas, contours_kept, -1, (0, 255, 0), 5) # Green line, increased thickness for visibility
        plt.subplot(1, 3, 1)
        plt.title(f"Step 4: Kept Contours ({len(contours_kept)})")
        plt.imshow(contour_canvas)
        plt.axis('off')
        plt.show()

    return candidates, im, contours_kept, mask


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
    
    # Apply the mask to the original image: pixels inside rects are kept, others become 0
    return cv2.bitwise_and(img, img, mask=mask)


def isolate_symbol(zone_crop, white_background=True, plot_debug=False, th_min=100, th_max=100000):
  
    if zone_crop is None or zone_crop.size == 0:
        return []

    hsv = cv2.cvtColor(zone_crop, cv2.COLOR_RGB2HSV)   
    
    if white_background:
        s = hsv[:, :, 1]
        _, mask = cv2.threshold(s, 90, 255, cv2.THRESH_BINARY) 
    else: 
        mask = cv2.inRange(hsv, (0, 0, 180), (179, 60, 255))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # RETR_CCOMP to have inner contours
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    

    im = zone_crop.copy()
    candidates = []
    contours_kept = []

    for i, cnt in enumerate(contours):
        cnt_smoothen = smooth_contour(cnt, sigma=4)

        rect_rotate = cv2.minAreaRect(cnt_smoothen) 
        w, h = rect_rotate[1]
        rect_area = w * h

        if rect_area > th_min and rect_area < th_max:
            box = np.int32(cv2.boxPoints(rect_rotate))
            cv2.drawContours(im, [box], 0, (0, 255, 0), 2)
            candidates.append(rect_rotate)
            contours_kept.append(cnt_smoothen) 

    if plot_debug:
        plt.figure(figsize=(24, 8)) 
        plt.subplot(1, 3, 1)
        plt.title("Original Zone Crop")
        plt.imshow(zone_crop)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Cleaned Mask (Step 3)")
        plt.imshow(mask, cmap='gray')
        plt.axis('off')

        plt.figure(figsize=(24, 8)) 
        contour_canvas = np.zeros_like(zone_crop)
        cv2.drawContours(contour_canvas, contours_kept, -1, (0, 255, 0), 5) # Green line, increased thickness for visibility
        plt.subplot(1, 3, 1)
        plt.title(f"Step 4: Kept Contours ({len(contours_kept)})")
        plt.imshow(contour_canvas)
        plt.axis('off')
        plt.show()

    return candidates, im, contours_kept, mask


def smooth_contour(cnt, sigma=3):
    cnt = cnt[:, 0, :]  # shape (N, 2)
    x = gaussian_filter1d(cnt[:, 0].astype(float), sigma)
    y = gaussian_filter1d(cnt[:, 1].astype(float), sigma)
    return np.stack([x, y], axis=1).reshape(-1, 1, 2).astype(np.int32)

def fill_contours(contours_kept, mask_shape):
    """Returns a binary mask with all contours filled."""
    mask = np.zeros_like(mask_shape, dtype=np.uint8)
    cv2.drawContours(mask, contours_kept, -1, 255, thickness=cv2.FILLED, lineType=cv2.LINE_8)
    return mask


def crop_filled_forms(image, contours, padding=50):
    """
    Returns a list of cropped images, one per contour,
    with only the filled symbol visible (background set to black).
    """
    crops = []

    # Padding so that we can take the bar under the 6 !!

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = w * h

        temp = image.copy()

        if rect_area > 10000:
            # Add padding (clamped to image bounds)
            x1 = max(x - padding, 0)
            y1 = max(y - padding, 0)
            x2 = min(x + w + padding, image.shape[1])
            y2 = min(y + h + padding, image.shape[0])

            """#Create a filled mask for this contour only
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

            # Apply mask to original image
            isolated = cv2.bitwise_and(image, image, mask=mask)"""

            # Crop to bounding box
            crop = temp[y1:y2, x1:x2]
            crops.append(crop)

    return crops