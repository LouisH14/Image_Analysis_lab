# Import main packages
from utils.lab_01_utils import *
from skimage.color import rgb2hsv
from skimage.morphology import (
    closing,
    opening,
    disk,
    remove_small_holes,
    remove_small_objects,
    binary_dilation,
)


##############################################################
##############################################################
### Part 1 : Segmentation

### 1:+ RGB
def extract_rgb_channels(img):
    """
    Extract RGB channels from the input image.

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.

    Return
    ------
    data_red: np.ndarray (M, N)
        Red channel of input image
    data_green: np.ndarray (M, N)
        Green channel of input image
    data_blue: np.ndarray (M, N)
        Blue channel of input image
    """

    # Get the shape of the input image
    M, N, _ = np.shape(img)

    # Define default values for RGB channels
    data_red = np.zeros((M, N))
    data_green = np.zeros((M, N))
    data_blue = np.zeros((M, N))

    data_red = img[:, :, 0]
    data_green = img[:, :, 1]
    data_blue = img[:, :, 2]

    return data_red, data_green, data_blue

def apply_rgb_threshold(img):
    """
    Apply threshold to input image.

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.

    Return
    ------
    img_th: np.ndarray (M, N)
        Thresholded image.
    """

    # Define the default value for the input image
    M, N, C = np.shape(img)
    img_th = np.zeros((M, N))

    # Use the previous function to extract RGB channels
    data_red, data_green, data_blue = extract_rgb_channels(img=img)

    data_red_th = data_red < 140
    data_green_th = data_green < 130
    data_blue_th = data_blue < 150
    img_th = data_red_th & data_green_th & data_blue_th

    return img_th


### 1:2 Other colorspaces
def extract_hsv_channels(img):
    """
    Extract HSV channels from the input image.

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.

    Return
    ------
    data_h: np.ndarray (M, N)
        Hue channel of input image
    data_s: np.ndarray (M, N)
        Saturation channel of input image
    data_v: np.ndarray (M, N)
        Value channel of input image
    """

    # Get the shape of the input image
    M, N, C = np.shape(img)

    # Define default values for HSV channels
    data_h = np.zeros((M, N))
    data_s = np.zeros((M, N))
    data_v = np.zeros((M, N))



    """"
    for i in range(M):
        for j in range(N):
            data_h[i, j] = hsv_img[i, j, 0]
            data_s[i, j] = hsv_img[i, j, 1]
            data_v[i, j] = hsv_img[i, j, 2]
    """
    # ------------------
    # Your code here 

    hsv_img = rgb2hsv(img)
    data_h = hsv_img[:, :, 0]
    data_s = hsv_img[:, :, 1]
    data_v = hsv_img[:, :, 2]
    
    # ------------------

    return data_h, data_s, data_v


def apply_hsv_threshold(img):
    """
    Apply threshold to the input image in hsv colorspace.

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.

    Return
    ------
    img_th: np.ndarray (M, N)
        Thresholded image.
    """

    # Define the default value for the input image
    M, N, C = np.shape(img)
    img_th = np.zeros((M, N))

    # Use the previous function to extract HSV channels
    data_h, data_s, data_v = extract_hsv_channels(img=img)

    # ------------------
    # Your code here ...
    # ------------------
    data_h_th = data_h < 0.8
    data_s_th = data_s > 0.3
    data_v_th = data_v < 0.6
    img_th = data_h_th & data_s_th & data_v_th

    return img_th


### 1.3 : Morphology
def apply_closing(img_th, disk_size):
    """
    Apply closing to input mask image using disk shape.

    Args
    ----
    img_th: np.ndarray (M, N)
        Image mask of size MxN.
    disk_size: int
        Size of the disk to use for opening

    Return
    ------
    img_closing: np.ndarray (M, N)
        Image after closing operation
    """

    # Define default value for output image
    img_closing = np.zeros_like(img_th)

    # ------------------
    # Your code here ...
    # ------------------
    img_closing = closing(img_th, disk(disk_size))

    return img_closing


def apply_opening(img_th, disk_size):
    """
    Apply opening to input mask image using disk shape.

    Args
    ----
    img_th: np.ndarray (M, N)
        Image mask of size MxN.
    disk_size: int
        Size of the disk to use for opening

    Return
    ------
    img_opening: np.ndarray (M, N)
        Image after opening operation
    """

    # Define default value for output image
    img_opening = np.zeros_like(img_th)

    # ------------------
    # Your code here ...
    # ------------------
    img_opening = opening(img_th, disk(disk_size))
    
    return img_opening

def remove_holes(img_th, size):
    """
    Remove holes from input image that are smaller than size argument.

    Args
    ----
    img_th: np.ndarray (M, N)
        Image mask of size MxN.
    size: int
        Minimal size of holes

    Return
    ------
    img_holes: np.ndarray (M, N)
        Image after remove holes operation
    """

    # Define default value for input image
    img_holes = np.zeros_like(img_th)

    # ------------------
    # Your code here ...
    # ------------------
    img_holes = remove_small_holes(img_th, size)

    return img_holes


def remove_objects(img_th, size):
    """
    Remove objects from input image that are smaller than size argument.

    Args
    ----
    img_th: np.ndarray (M, N)
        Image mask of size MxN.
    size: int
        Minimal size of objects

    Return
    ------
    img_obj: np.ndarray (M, N)
        Image after remove small objects operation
    """

    # Define default value for input image
    img_obj = np.zeros_like(img_th)

    # ------------------
    # Your code here ...
    # ------------------
    img_obj = remove_small_objects(img_th, size)
    return img_obj


def apply_morphology(img_th):
    """
    Apply morphology to thresholded image

    Args
    ----
    img_th: np.ndarray (M, N)
        Image mask of size MxN.

    Return
    ------
    img_morph: np.ndarray (M, N)
        Image after morphological operations
    """

    img_morph = np.zeros_like(img_th)

    # ------------------
    # Your code here ...
    # ------------------
    
    img_morph = opening(img_th, disk(2))
    img_morph = closing(img_morph, disk(3))
    img_morph = remove_small_holes(img_morph, 500)
    img_morph = remove_small_objects(img_morph, 500)
    

    return img_morph


### 1.4: Region Growing
def region_growing(seeds: list[tuple], img: np.ndarray, n_max: int = 10, **kwargs):
    """
    Run region growing on input image using seed points.

    Args
    ----
    seeds: list of tuple
        List of seed points
    img: np.ndarray (M, N, C)
        RGB image of size M, N, C
    n_max: int
        Number maximum of iterations before stopping algorithm

    Return
    ------
    rg: np.ndarray (M, N)
        Image after region growing has been performed
    """

    M, N, _ = img.shape
    rg = np.zeros((M, N)).astype(bool)

    # ------------------
    # Your code here ...
    # ------------------
    thresh_r = kwargs.get('thresh_r')
    thresh_g = kwargs.get('thresh_g')
    thresh_b = kwargs.get('thresh_b')
    
    rg[tuple(np.array(seeds).T)] = True

    for i in range(n_max):
        rg_dilated = binary_dilation(rg)
        rg_new = rg_dilated & (rg == False) # garde juste les nveaux
        rg_new = rg_new & (img[:, :, 0] < thresh_r) & (img[:, :, 1] < thresh_g) & (img[:, :, 2] < thresh_b)
        if np.sum(rg_new) == 0:
            break
        rg = rg | rg_new
   

    return rg


##############################################################
##############################################################
### Part 2 : 

##############################################################
##############################################################
### Part 3 : 