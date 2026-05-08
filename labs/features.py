import numpy as np
import cv2 # Added for SIFT
import matplotlib.pyplot as plt
from core import image


# =============================================================================
# FAMILY 2: LOCAL FEATURE EXTRACTION (SIFT & ORB)
# =============================================================================

def apply_sift(image_np: np.ndarray):
    """
    Applies the SIFT algorithm to a given image to detect keypoints and compute descriptors.

    Args:
        image_np (np.ndarray): The input image as a NumPy array (e.g., from the 'image' class).
                               It can be RGB or grayscale.

    Returns:
        tuple: A tuple containing:
            - keypoints (list): A list of cv2.KeyPoint objects.
            - descriptors (np.ndarray): A NumPy array of SIFT descriptors.
                                        Returns None if no keypoints are found.
    """
    # Convert image to grayscale if it's not already
    if len(image_np.shape) == 3:
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image_np

    # Initialize SIFT detector and detect keypoints and compute descriptors
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    return keypoints, descriptors

def visualize_sift(im_obj: image):
    """
    Helper to visualize SIFT keypoints on a specific image object.
    """
    img = im_obj.get()
    kp, des = apply_sift(img)
    
    # Draw keypoints. DRAW_RICH_KEYPOINTS draws circles representing 
    # the size and orientation of the feature.
    img_kp = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(img_kp)
    plt.title(f"SIFT Features: Image {im_obj.image_number} ({len(kp)} points)")
    plt.axis('off')
    plt.show()

def TEST_sift_quality(num_samples=3):
    """
    Iterates through a few images to visually inspect SIFT output.
    """
    print(f"Checking SIFT output for {num_samples} samples...")
    for i in range(num_samples):
        im = image(i)
        visualize_sift(im)
        _, des = apply_sift(im.get())
        if des is not None:
            print(f"Sample {i}: Descriptors shape {des.shape}")
    print("Inspection complete.")


    
def apply_orb(image_np: np.ndarray):
    if len(image_np.shape) == 3:
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image_np

    # Initialize ORB detector. By default, it detects up to 500 features.
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray_image, None)
    return keypoints, descriptors

def visualize_orb(im_obj: image):
    """
    Helper to visualize ORB keypoints on a specific image object.
    """
    img = im_obj.get()
    kp, des = apply_orb(img)
    
    # Draw keypoints
    img_kp = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(img_kp)
    plt.title(f"ORB Features: Image {im_obj.image_number} ({len(kp)} points)")
    plt.axis('off')
    plt.show()
    return kp, des


def TEST_orb_quality(num_samples=3):
    """
    Iterates through a few images to visually inspect ORB output.
    """
    print(f"Checking ORB output for {num_samples} samples...")
    for i in range(num_samples):
        im = image(i)
        _, des = visualize_orb(im)
        if des is not None:
            print(f"Sample {i}: Descriptors shape {des.shape}")
    print("Inspection complete.")



