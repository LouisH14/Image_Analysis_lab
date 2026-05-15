import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import core
import zones
from skimage.color import rgb2hsv
from scipy.ndimage import gaussian_filter1d

TH_ISOLATE_CARDE_WHITEBCK = 75000
TH_ISOLATE_CARDE_NOISYBCK = 220000

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

    im_with_boxes = zone_crop.copy() # To store the image with drawn boxes

    if white_background:

        # for the colored cards
        s = hsv[:, :, 1]
        _, mask_colored = cv2.threshold(s, 90, 255, cv2.THRESH_BINARY)

        # for dark template cards
        mean = im_with_boxes.mean(axis=2)
        mask_black = (mean < 100) 
        mask_black = mask_black.astype(np.uint8) * 255

        # Combinaison des deux masques (UNION)
        mask = cv2.bitwise_or(mask_colored, mask_black)

        # Step 3: Morphological Cleanup - bridge gaps in the detected border
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Step 4: Contour Detection
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Step 5 & 6: Filtering and Candidate Extraction
        candidates = []
        contours_kept = []

        for i, cnt in enumerate(contours):
            rect_rotate = cv2.minAreaRect(cnt) 
            w, h = rect_rotate[1]
            rect_area = w * h
            if rect_area > threshold:
                box = np.int32(cv2.boxPoints(rect_rotate))
                cv2.drawContours(im_with_boxes, [box], 0, (0, 255, 0), 2) # Draw on im_with_boxes
                candidates.append(rect_rotate)
                contours_kept.append(cnt) # to discard contours attached to cards from other players (that appear on the border of the region)
                
        
    """else: # the background has colorized patterns
        # On patterned backgrounds, white card borders have low saturation and high value
        mask = cv2.inRange(hsv, (0, 0, 180), (179, 60, 255))

        # Step 3: Morphological Cleanup - bridge gaps in the detected border
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Step 4: Contour Detection
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #contours = keep_rectangular_contours(contours, epsilon_factor=0.9, min_area=1)

        # Step 5 & 6: Filtering and Candidate Extraction
        candidates = []
        contours_kept = []

        for i, cnt in enumerate(contours):
            rect_rotate = cv2.minAreaRect(cnt) 
            w, h = rect_rotate[1]
            rect_area = w * h
            if rect_area > TH_ISOLATE_CARDE_WHITEBCK and rect_area < threshold:
                box = np.int32(cv2.boxPoints(rect_rotate))
                cv2.drawContours(im_with_boxes, [box], 0, (0, 255, 0), 2) # Draw on im_with_boxes
                candidates.append(rect_rotate)
                contours_kept.append(cnt) # to discard contours attached to cards from other players (that appear on the border of the region) 
                """

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
    
    return candidates, im_with_boxes, contours_kept, mask # Return im_with_boxes instead of im


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


def crop_cards_from_zone(zone_to_crop):
    # zone_to
    gray = cv2.cvtColor(zone_to_crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cards = []
    for cnt in contours:
        x, y, w, h = [int(v) for v in cv2.boundingRect(cnt)]
        cards.append(zone_to_crop[y:y+h, x:x+w])

    return cards


def isolate_symbol(zone_crop, white_background=True, plot_debug=False, th_min=100, th_max=100000):
  
    if zone_crop is None or zone_crop.size == 0:
        return []

    hsv = cv2.cvtColor(zone_crop, cv2.COLOR_RGB2HSV)
    s=hsv[:, :, 1]

    im_with_boxes = zone_crop.copy() # To store the image with drawn boxes

    if white_background:
        _, mask = cv2.threshold(s, 90, 255, cv2.THRESH_BINARY) 
    else: 
        mask = cv2.inRange(hsv, (0, 0, 180), (179, 60, 255))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # RETR_CCOMP to have inner contours
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    

    candidates = []
    contours_kept = []

    for i, cnt in enumerate(contours):
        cnt_smoothen = smooth_contour(cnt, sigma=4)
        rect_rotate = cv2.minAreaRect(cnt_smoothen) 
        w, h = rect_rotate[1]
        rect_area = w * h
        if rect_area > th_min and rect_area < th_max:
            box = np.int32(cv2.boxPoints(rect_rotate))
            cv2.drawContours(im_with_boxes, [box], 0, (0, 255, 0), 2) # Draw on im_with_boxes
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
    
    return candidates, im_with_boxes, contours_kept, mask # Return im_with_boxes instead of im


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
    print(len(contours))
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

            """Create a filled mask for this contour only
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

            # Apply mask to original image
            isolated = cv2.bitwise_and(image, image, mask=mask)"""

            # Crop to bounding box
            crop = temp[y1:y2, x1:x2]
            crops.append(crop)

    return crops

def keep_rectangular_contours(contours, epsilon_factor=0.05, min_area=1000):
    """Keep only contours that approximate to a rectangle (4 vertices)."""
    rectangular_cnt = []
    
    for cnt in contours:
        # Skip tiny contours
        if cv2.contourArea(cnt) < min_area:
            continue
        
        # Approximate the contour to a polygon
        epsilon = epsilon_factor * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # A rectangle has 4 vertices
        if len(approx) == 4:
            rectangular_cnt.append(cnt)
    
    return rectangular_cnt


#########################################################################
########################### TESTS #######################################
#########################################################################

def test_white_nooverlapp(nb):
    im_obj = core.image(core.valid_nb(nb))
    original_im = im_obj.get()
    zone1, zone2, zone3, zone4, zone0 = zones.extract_zones(original_im)
    all_zones = [zone0, zone1, zone2, zone3, zone4] # Center first, then players 1-4
    zone_names = ["Center", "Player 1", "Player 2", "Player 3", "Player 4"]

    all_zone_crops_for_plotting = []
    zones_with_crops_count = 0

    for i, zone in enumerate(all_zones):
        # Ensure zone is not empty before processing
        if zone is None or zone.size == 0:
            print(f"Zone {zone_names[i]} is empty, skipping.")
            continue

        # Step 1: Isolate cards (rectangles) within the zone
        # Set plot_debug=False to avoid intermediate plots from isolate_cards
        candidates_cards, im_cards_drawn, _, _ = isolate_cards(zone, white_background=True, plot_debug=False, threshold=75000)

        if not candidates_cards:
            print(f"No cards found in {zone_names[i]}.")
            continue

        # Step 2: Mask the zone to keep only the detected card areas
        zone_masked = mask_rectangles(zone, candidates_cards)

        # Step 3: Isolate symbols within the masked card areas
        # Set plot_debug=False to avoid intermediate plots from isolate_symbol
        _, im_symbols_drawn, contours_kept_symbols, mask_symbols = isolate_symbol(zone_masked, white_background=True, plot_debug=False, th_min=1900, th_max=40000)

        if not contours_kept_symbols:
            print(f"No symbols found in {zone_names[i]}.")
            continue

        # Step 4: Fill the detected symbol contours
        im_filled_contours = fill_contours(contours_kept_symbols, mask_symbols)

        # Collect the filled mask for this zone for vertical stacking
        all_zone_crops_for_plotting.append((zone_names[i], im_filled_contours))
        zones_with_crops_count += 1

    if not all_zone_crops_for_plotting:
        print(f"No symbols found for image {nb} across all zones.")
        return 0

    # Spacing for original image (span 2 column units)
    orig_span = 2
    total_cols = orig_span + 1 # Original span + 1 column for the vertical stack

    # Create the figure with GridSpec, stacking zones closer by reducing height multiplier and setting hspace
    fig = plt.figure(figsize=(12, zones_with_crops_count * 2))
    gs = fig.add_gridspec(zones_with_crops_count, total_cols, hspace=0.3)
    fig.suptitle(f"Image {nb}: Original and Filled Symbol Masks per Zone", fontsize=16)

    # Plot the original image on the left, spanning all rows
    ax_orig = fig.add_subplot(gs[:, :orig_span])
    ax_orig.imshow(original_im)
    ax_orig.set_title("Original Image", fontsize=14)
    ax_orig.axis('off')

    # Plot each zone's filled mask in a vertical stack on the right
    for row_idx, (zone_name, mask_img) in enumerate(all_zone_crops_for_plotting):
        ax = fig.add_subplot(gs[row_idx, orig_span])
        ax.imshow(mask_img, cmap='gray')
        ax.set_title(zone_name, fontsize=12)
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent suptitle overlap
    plt.show()

    return 0


def test_isolate_cards(nb, threshold=75000):
    im_obj = core.image(core.valid_nb(nb))
    original_im = im_obj.get()

    zone1, zone2, zone3, zone4, zone0 = zones.extract_zones(original_im)
    all_zones = [zone0, zone1, zone2, zone3, zone4]

    all_isolated_cards = []
    for zone in all_zones:
        candidates, _, _, _ = isolate_cards(zone, white_background=True, plot_debug=False, threshold=threshold)
        zone_masked = mask_rectangles(zone, candidates)

        one_isolated_card = crop_cards_from_zone(zone_masked) # zone_masked is converted into gray scal

        all_isolated_cards.append(one_isolated_card)

    # Display
    fig, axes = plt.subplots(1 + len(all_isolated_cards), 5, figsize=(10, 10))

    # Original im
    axes[0, 0].imshow(original_im)
    axes[0, 0].axis('off')
    for j in range(1, 5):
        axes[0, j].axis('off')

    # Cartes by zones
    for i, zone_cards in enumerate(all_isolated_cards):
        for j in range(5):
            ax = axes[i + 1, j]
            if j < len(zone_cards):
                ax.imshow(zone_cards[j], cmap='gray')
            ax.axis('off')

    plt.tight_layout()
    plt.show()

    return 0