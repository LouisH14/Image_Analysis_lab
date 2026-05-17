import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

def order_points(pts):
    """
    Orders 4 points into a consistent clockwise order: 
    [top-left, top-right, bottom-right, bottom-left].
    
    This consistency ensures that the detected card coordinates align 
    correctly with the normalized reference space (200x300 rectangle).

    Args:
        pts (np.ndarray): An array of four points (x, y) coordinates, 
                          usually floating point from cv2.boxPoints().

    Returns:
        np.ndarray: A (4, 2) array of sorted floating point coordinates.
    """
    # Initialize the rectangle array
    rect = np.zeros((4, 2), dtype="float32")

    # Top-left has the smallest sum (x + y), bottom-right has the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Top-right has the smallest difference (y - x)
    # Bottom-left has the largest difference (y - x)
    diff = pts[:, 1] - pts[:, 0]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def create_normalized_mask(mask_contours, mask_corners, ref_size=(200, 300), ref_Y=2662, ref_X=4000):
    """
    Generates a 200x300 normalized mask from a contour defined in 4000x2662 image space.
    
    Args:
        mask_contours (list): The internal color patches you want to extract.
        mask_corners (np.ndarray): The 4 corners of the WHOLE card (the border).
                                   Using the card border as the reference ensures 
                                   the internal patches are mapped proportionally.
        ref_size (tuple): Target dimensions (width, height).

    Returns:
        np.ndarray: A 200x300 binary mask (uint8).
    """
    # 1. Create a temporary full-size mask in image space
    # Using the standard dimensions from your project (Height: ref_Y, Width: ref_X)
    temp_mask = np.zeros((ref_Y, ref_X), dtype="uint8")
    cv2.drawContours(temp_mask, mask_contours, -1, 255, thickness=cv2.FILLED)

    # 2. Compute transform from Image Space -> Normalized Space
    src_pts = order_points(mask_corners)
    dst_pts = np.array([
        [0, 0], [ref_size[0], 0], [ref_size[0], ref_size[1]], [0, ref_size[1]]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # 3. Warp the image-space mask into the 200x300 space
    normalized_mask = cv2.warpPerspective(temp_mask, M, ref_size, flags=cv2.INTER_NEAREST)
    
    return normalized_mask

def warp_normalized_mask(card_corners, normalized_mask, full_image_shape):
    """
    Maps a normalized mask to the card's actual position in the image.

    Args:
        card_corners (np.ndarray): The 4 corner points from cv2.boxPoints.
        normalized_mask (np.ndarray): The 200x300 mask from your reference space.
        full_image_shape (tuple): The (height, width) of the source image.

    Returns:
        np.ndarray: A binary mask the size of the full image with the mask superimposed.
    """
    # Step 1: Sort detected corners into Top-Left, Top-Right, Bottom-Right, Bottom-Left
    dst_pts = order_points(card_corners)

    # Step 2: Define the 4 corners of the normalized reference rectangle
    # Order must match Step 1: TL, TR, BR, BL
    ref_w, ref_h = 200, 300
    src_pts = np.array([
        [0, 0],           # Top-Left
        [ref_w, 0],       # Top-Right
        [ref_w, ref_h],   # Bottom-Right
        [0, ref_h]        # Bottom-Left
    ], dtype="float32")

    # Step 3: Compute the perspective transform
    # M maps points from Normalized Space -> Image Space
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Step 4: Warp the normalized mask into image space
    # We use the full image dimensions for the output
    h, w = full_image_shape[:2]
    
    # INTER_NEAREST is used to keep the mask binary (0 or 255)
    warped_mask = cv2.warpPerspective(
        normalized_mask, M, (w, h), flags=cv2.INTER_NEAREST
    )

    return warped_mask

def get_mean_color(image, mask):
    """
    Computes the mean BGR color of the pixels in the image using a binary mask.

    Args:
        image (np.ndarray): The source BGR image.
        mask (np.ndarray): The warped binary mask from Step 4.

    Returns:
        tuple: The mean (Blue, Green, Red) color components.
    """

    bright_mask = np.all(image > 200, axis=2)  # True if 3 channels >200
    exclude_mask = (~bright_mask).astype(np.uint8) * 255  # 8just invert so keep only non-shite pixels

    combined_mask = cv2.bitwise_and(mask, exclude_mask)
    mean_values = cv2.mean(image, mask=combined_mask)
    return mean_values[:3] # cv2.mean returns a 4-element tuple (B, G, R, alpha)

def classify_color(mean_bgr):
    """
    Classifies the card color by comparing the measured BGR mean to reference colors.

    Args:
        mean_bgr (tuple): The measured average color (B, G, R).

    Returns:
        str: The predicted UNO color ("RED", "BLUE", "GREEN", "YELLOW", or "BLACK").
    """
    # Reference BGR values for typical UNO card colors. 
    # These may require fine-tuning based on your specific camera's white balance.
    reference_colors = {
        "RED":    np.array([144, 157, 240]),
        "BLUE":   np.array([200, 50, 50]),
        "GREEN":  np.array([100, 190, 120]),
        "YELLOW": np.array([50, 210, 250]),
        "BLACK":  np.array([120, 110, 107])
    }

    measured = np.array(mean_bgr)
    best_match = None
    min_distance = float('inf')

    for color_name, ref_bgr in reference_colors.items():
        distance = np.linalg.norm(measured - ref_bgr)
        if distance < min_distance:
            min_distance = distance
            best_match = color_name

    return best_match


def get_corners_from_contours(contours):
    if contours is None or len(contours) == 0:
        return None
    if isinstance(contours, np.ndarray):
        all_pts = contours  # déjà un seul array
    else:
        all_pts = np.vstack(contours)  # liste de contours
    rect = cv2.minAreaRect(all_pts)
    box = np.round(cv2.boxPoints(rect)).astype(np.int32)
    return order_points(box)


def run_visual_test(mask_contours, mask_corners, ref_X, ref_Y):
    """
    A visual test function to verify that masks are correctly normalized 
    and superimposed back onto image space.
    """
    # 1. Create the normalized template (200x300)
    # This simulates extracting the template from a reference card image
    norm_mask = create_normalized_mask(
        mask_contours, 
        mask_corners, 
        ref_size=(200, 300), 
        ref_Y=ref_Y, 
        ref_X=ref_X
    )
    # 2. Warp it back to the full image space
    # This simulates projecting the template onto a detected card
    warped_back = warp_normalized_mask(mask_corners, norm_mask, (ref_Y, ref_X))

    # 3. Plot the results for comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Draw what the input looked like
    input_canvas = np.zeros((ref_Y, ref_X), dtype="uint8")
    #cv2.drawContours(input_canvas, mask_contours, -1, 255, thickness=cv2.FILLED)
    # Draw the mask_corners on the input for clarity
    sorted_mask_corners = order_points(mask_corners)
    poly = patches.Polygon(sorted_mask_corners, linewidth=2, edgecolor='r', facecolor='none')
    
    axes[0].imshow(input_canvas, cmap='gray')
    axes[0].add_patch(poly)
    axes[0].set_title("Input (Red box = mask_corners)")
        
    axes[1].imshow(norm_mask, cmap='gray')
    axes[1].set_title("Template (Correctly Centered)")
    
    axes[2].imshow(warped_back, cmap='gray')
    axes[2].set_title("Superimposed Result")

    
    plt.tight_layout()
    plt.show()

def test_on_real_data(img_with_card, target_corners, mask_contours):
    """
    Complete test pipeline using real data parameters.
    
    Args:
        target_img (np.ndarray): The image containing the card to identify.
        target_corners (np.ndarray): 4 corners of the detected card in target_img.
        mask_contours (list): Contours of the color regions from the reference image.
        mask_corners (np.ndarray): 4 corners of the reference card.
        ref_X (int): Width of the reference image.
        ref_Y (int): Height of the reference image.
        
    Returns:
        str: The identified color.
    """
    print("\n--- Running Real Data Pipeline Test ---")
    target_img = img_with_card.copy()
    
    # 1. Normalize the mask using the reference card data
    # This creates the standard 200x300 template
    print("1. Creating normalized template...")
    mask_corners = get_corners_from_contours(mask_contours)
    ref_Y = img_with_card.shape[0]
    ref_X = img_with_card.shape[1]
    norm_mask = create_normalized_mask(
        mask_contours, 
        mask_corners, 
        ref_size=(200, 300), 
        ref_Y=ref_Y, 
        ref_X=ref_X
    )
    
    # 2. Project the mask onto the new detected card
    print("2. Superimposing mask on target box...")
    warped_mask = warp_normalized_mask(target_corners, norm_mask, target_img.shape)
    
    # 3. Compute mean color of the pixels under the mask
    print("3. Computing mean color...")
    mean_bgr = get_mean_color(target_img, warped_mask)
    print(f"   Measured BGR: {mean_bgr}")
    
    # 4. Classify the color
    print("4. Classifying...")
    color_result = classify_color(mean_bgr)
    print(f"   Result: {color_result}")

    # 5. Visualization of the mask application
    vis_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    # Find the contours of the warped mask to draw them on the target image
    mask_contours_vis, _ = cv2.findContours(warped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis_img, mask_contours_vis, -1, (0, 255, 0), 3) # Green contour
    
    plt.figure(figsize=(12, 8))
    plt.imshow(vis_img)
    plt.title(f"Target Image with Applied Mask (Predicted: {color_result})")
    plt.axis('off')
    plt.show()
    
    return color_result, norm_mask
