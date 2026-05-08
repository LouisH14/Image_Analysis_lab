# Import main packages
from utils.lab_02_utils import *
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    closing,
    disk,
    opening,
)
from skimage.transform import rotate, resize
from sklearn.metrics.pairwise import euclidean_distances
from skimage.measure import regionprops

import cv2
import numpy as np


###########################################################################
###########################################################################
### Part 1 : Preprocessing


### 1.1 : Selection

def extract_label(images: np.ndarray, labels: np.ndarray, target_label: int):
    """
    The function returns only the images that have target_label as labels.

    Args
    ----
    images: np.ndarray (N, 28, 28)
        Source images - handwritten digits
    labels: np.ndarray (N)
        List of labels associated with the input image
    target_label: int
        Selected target label

    Return
    ------
    img_extract: np.ndarray (M, 28, 28)
        Extracted images that have target_label as label (M should be lower than N).
    """
    n, d, _ = np.shape(images)
    M = np.sum(labels == target_label)

    img_extract2 = np.zeros((M, d, d))
    idx = 0
    for k in range(n):
        if labels[k] == target_label:
            img_extract2[idx] = images[k]
            idx += 1
    # ------------------
    # Your code here ...
    # ------------------

    img_extract = images[labels == target_label]

    return img_extract


### 1.2: Preprocessing
def plot_histogram(image: np.ndarray, bins: int = 256):
    """
    Plot histogram for one given image.

    Args
    ----
    image: np.ndarray (H, W)
        One grayscale image.
    bins: int
        Number of histogram bins.

    Return
    ------
    hist: np.ndarray (bins,)
        Histogram counts.
    bin_edges: np.ndarray (bins + 1,)
        Histogram bin edges.
    """
    import matplotlib.pyplot as plt

    img = image.astype(np.float32).ravel()
    hist, bin_edges = np.histogram(img, bins=bins, range=(0, 256))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    plt.figure(figsize=(8, 4))
    plt.scatter(bin_centers, hist, lw=2, marker="+")
    plt.title("Histogram of one image")
    plt.xlabel("Pixel intensity")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 255)
    plt.show()

    print(f"Total count = {hist.sum()} | Number of pixels = {image.size}")

    return hist, bin_edges



def preprocess(images: np.ndarray):
    """
    Apply the processing step to images to achieve better data uniformity.

    Args
    ----
    images: np.ndarray (N, 28, 28)
        Source images

    Return
    ------
    img_process: np.ndarray (N, 28, 28)
        Processed images.
    """

    # Get the shape of input data and set dummy values
    n, d, _ = np.shape(images)
    img_process = np.zeros_like(images, dtype=np.uint8)

    img_process = (images > 100).astype(np.uint8)
    
    
    for i in range(len(img_process)):
        img_process[i] = closing(img_process[i], disk(1))
    
    pos = [9, 13]
    largeur = 3
    v = 1
    for i in range(largeur):
        for j in range(largeur):
            img_process[3][pos[0]+i, pos[1]+j] = v
    
    return img_process







###########################################################################
###########################################################################
### Part 2 : Fourier Descriptors



### 2.1: Get contour and descriptors
def find_contour(images: np.ndarray):
    """
    Find the contours for the set of images

    Args
    ----
    images: np.ndarray (N, 28, 28)
        Source images to process

    Return
    ------
    contours: list of np.ndarray
        List of N arrays containing the coordinates of the contour. Each element of the
        list is an array of 2d coordinates (K, 2) where K depends on the number of elements
        that form the contour.
    """

    # Get number of images to process
    N, _, _ = np.shape(images)
    # Fill in dummy values (fake points)
    contours = [np.array([[0, 0], [1, 1]]) for i in range(N)]
    for i in range(N):
        contours[i], _ = cv2.findContours(
            images[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours[i] = contours[i][0].squeeze()  # Get the first contour and remove extra dimensions
    # ------------------
    # Your code here ...
    # ------------------

    return contours


def compute_descriptor_padding(contours: np.ndarray, n_samples: int = 11):
    """
    Compute Fourier descriptors of input images

    Args
    ----
    contours: list of np.ndarray
        List of N arrays containing the coordinates of the contour. Each element of the
        list is an array of 2d coordinates (K, 2) where K depends on the number of elements
        that form the contour.
    n_samples: int
        Number of samples to consider. If the contour length is higher, discard the remaining part. If it is shorter, add padding.
        Make sure that the first element of the descriptor represents the continuous component.

    Return
    ------
    descriptors: np.ndarray complex (N, n_samples)
        Computed complex Fourier descriptors for the given input images
    """

    N = len(contours)
    descriptors = np.zeros((N, n_samples), dtype=np.complex128)

    for i in range(N):
        cnt = np.asarray(contours[i])

        # Skip malformed or empty contours and keep zero descriptor.
        if cnt.ndim != 2 or cnt.shape[0] == 0 or cnt.shape[1] < 2:
            continue

        # Keep only x,y coordinates and enforce fixed length using truncate/padding.
        cnt_xy = cnt[:, :2]
        if cnt_xy.shape[0] >= n_samples:
            cnt_fixed = cnt_xy[:n_samples]
        else:
            pad = np.zeros((n_samples - cnt_xy.shape[0], 2), dtype=cnt_xy.dtype)
            cnt_fixed = np.vstack((cnt_xy, pad))

        # Convert contour points to complex sequence and apply FFT.
        signal = cnt_fixed[:, 0].astype(np.float64) + 1j * cnt_fixed[:, 1].astype(
            np.float64
        )
        descriptors[i] = np.fft.fft(signal)

    return descriptors


def linear_interpolation(contours: np.ndarray, n_samples: int = 11):
    """
    Perform interpolation/resampling of the contour across n_samples.

    Args
    ----
    contours: list of np.ndarray
        List of N arrays containing the coordinates of the contour. Each element of the
        list is an array of 2d coordinates (K, 2) where K depends on the number of elements
        that form the contour.
    n_samples: int
        Number of samples to consider along the contour.

    Return
    ------
    contours_inter: np.ndarray (N, n_samples, 2)
        Interpolated contour with n_samples
    """

    N = len(contours)
    contours_inter = np.zeros((N, n_samples, 2))
    for i in range(N):
        cnt = np.asarray(contours[i])

        # Skip malformed or empty contours and keep zero descriptor.
        if cnt.ndim != 2 or cnt.shape[0] == 0 or cnt.shape[1] < 2:
            continue

        # Keep only x,y coordinates and perform linear interpolation.
        cnt_xy = cnt[:, :2]
        x = cnt_xy[:, 0]
        y = cnt_xy[:, 1]
        
        """
        # intuitive step t
        t = np.arange(len(cnt_xy))
        t_inter = np.linspace(0, len(cnt_xy) - 1, n_samples)
        """
        
        # step with the cumulated Euclidean Distance:
        diff = np.diff(cnt_xy, axis=0)
        t = np.concatenate([[0], np.cumsum(np.sqrt((diff**2).sum(axis=1)))])
        t_inter = np.linspace(0, t[-1], n_samples)

        x_inter = np.interp(t_inter, t, x)
        y_inter = np.interp(t_inter, t, y)
        contours_inter[i] = np.stack((x_inter, y_inter), axis=-1)

    # ------------------
    # Your code here ...
    # ------------------

    return contours_inter




### 2.2 : Reconstruction
def compute_reverse_descriptor(descriptor: np.ndarray, n_samples: int = 11):
    """
    Reverse a Fourier descriptor to xy coordinates given a number of samples.

    Args
    ----
    descriptor: np.ndarray (D,)
        Complex descriptor of length D.
    n_samples: int
        Number of samples to consider to reverse transformation.

    Return
    ------
    x: np.ndarray complex (n_samples,)
        x coordinates of the contour
    y: np.ndarray complex (n_samples,)
        y coordinates of the contour
    """

    x = np.zeros(n_samples)
    y = np.zeros(n_samples)

    # ------------------
    # Your code here ...
    # ------------------

    # Size normalisation (truncate/pad) as before:
    d = np.asarray(descriptor, dtype=np.complex128).flatten()
    if len(d) >= n_samples:
        d_fixed = d[:n_samples]
    else:
        d_fixed = np.zeros(n_samples, dtype=np.complex128)
        d_fixed[: len(d)] = d

    # iFFT
    contour = np.fft.ifft(d_fixed, n=n_samples)

    # separate x and y
    x = np.real(contour)
    y = np.imag(contour)

    return x, y


### 2.3 : Invariance

def apply_rotation(img: np.ndarray):
    """
    Apply random rotation to input the image

    Args
    ----
    image: np.ndarray (28, 28)
        Source images

    Return
    ------
    rotated: np.ndarray (28, 28)
        Rotated source images
    """

    rotated = np.zeros_like(img)
    random_angle = np.random.uniform(0, 360)
    # random_angle = np.random.choice([0, 90, 180, 270])
    rotation_matrix = cv2.getRotationMatrix2D((14, 14), random_angle, 1)
    rotated = cv2.warpAffine(img, rotation_matrix, (28, 28), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue = 0)

    # ------------------
    # Your code here ...
    # ------------------

    return rotated


def apply_scaling(img: np.ndarray):
    """
    Apply random scaling to input image
git 
    Args
    ----
    image: np.ndarray (28, 28)
        Source images

    Return
    ------
    scaled: np.ndarray (28, 28)
        Scaled source images
    """

    scaled = np.zeros_like(img)
    random_scale = np.random.uniform(0.5, 1.3)
    scale_matrix = cv2.getRotationMatrix2D((14, 14), 0, random_scale)
    scaled = cv2.warpAffine(img, scale_matrix, (28, 28), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # ------------------
    # Your code here ...
    # ------------------

    return scaled


def apply_translate(img: np.ndarray):
    """
    Apply random x and y translation to input image

    Args
    ----
    image: np.ndarray (28, 28)
        Source images

    Return
    ------
    translated: np.ndarray (28, 28)
        Translated source images
    """

    translated = np.zeros_like(img)
    
    random_tx = np.random.uniform(-5, 5)
    random_ty = np.random.uniform(-5, 5)

    translation_matrix = np.float32([[1, 0, random_tx], [0, 1, random_ty]])
    translated = cv2.warpAffine(img, translation_matrix, (28, 28), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # ------------------
    # Your code here ...
    # ------------------

    return translated


def translation_invariant(features):
    """
    Make input Fourier descriptors invariant to translation.

    Args
    ----
    features: np.ndarray (N, D)
        The Fourier descriptors of N images over D features.

    Return
    ------
    features_inv: np.ndarray (N, K)
        The Fourier descriptors invariant to translation of N images
        over K (K <= N) features.
    """

    # Set default values
    features_inv = np.zeros_like(features)

    # ------------------
    # Your code here ...
    # ------------------

    N, D = features.shape

    for i in range(N):
        features_inv[i] = features[i]
        features_inv[i, 0] = 0

    return features_inv


def rotation_invariant(features):
    """
    Make input Fourier descriptors invariant to rotation.

    Args
    ----
    features: np.ndarray (N, D)
        The Fourier descriptors of N images over D features.

    Return
    ------
    features_inv: np.ndarray (N, K)
        The Fourier descriptors invariant to rotation of N images
        over K (K <= N) features.
    """

    # Set default values
    features_inv = np.zeros_like(features)

    # ------------------
    # Your code here ...
    # ------------------        
     
    N, D = features.shape
    for i in range(N):

        features_inv[i] = abs(features[i])

    return features_inv



def scaling_invariant(features):
    """
    Make input Fourier descriptors invariant to scaling.

    Args
    ----
    features: np.ndarray (N, D)
        The Fourier descriptors of N images over D features.

    Return
    ------
    features_inv: np.ndarray (N, K)
        The Fourier descriptors invariant to scaling of N images
        over K (K <= N) features.
    """

    # Set default values
    features_inv = np.zeros_like(features)

    # ------------------
    # Your code here ...
    # ------------------

    N, D = features.shape

    for i in range(N): 

        ratio = np.abs(features[i, 1]) # coeff0 refers to the centroid, coeff 1 refers to the global shape
        if ratio < 1e-8:
            ratio = 1e-8

        features_inv[i] = features[i] / ratio

    return features_inv



###########################################################################
###########################################################################
### Part 3 : Other descriptors


### 3.1 : Distance map
def reference_pattern(imgs):
    """
    Compute the reference pattern for a given set of images. The reference pattern
    is estimated as the average of all images of the same pattern.

    Args
    ----
    imgs: np.ndarray (N, 28, 28)
        Source images

    Return
    ------
    pattern: np.ndarray (28, 28)
        Thresholded reference pattern that is the average of all shapes.
    """

    # Initialize pattern
    pattern = np.zeros((imgs[0].shape[0], imgs[0].shape[1])) # 28x28

    # ------------------
    # Your code here ...
    # ------------------

    pattern = np.mean(imgs, axis=0)
    
    return pattern

def compute_distance_map(pattern: np.ndarray):
    """
    Compute the distance map for the given pattern. The values of the map are computed as
    the distance to the closest pattern contour.

    Args
    ----
    pattern: np.ndarray (28, 28)
        Pattern to process

    Return
    ------
    distance_map: np.ndarray (28, 28)
        Distance map where each entry is the distance to the closest pattern contour (shortest
        distance to pattern)
    """

    # Initialize dummy values
    distance_map = np.zeros_like(pattern)

    # ------------------
    # Your code here ...
    # ------------------

    inf = 2*28*4
    epsilon = 0.2
    distance_map = np.where(pattern > (np.mean(pattern[pattern != 0])+epsilon), 0, inf).astype(np.uint8)
    adj = 3
    hyp = 4
    #plt.imshow(distance_map)

    X, Y = np.shape(distance_map)

    # Aller 
    for y in range(0, Y): # itère de 1 à (X-1) inclus
        for x in range(0, X):
            d = distance_map[x-1, y-1] if x!=0 and y!=0 else inf
            h = distance_map[x, y-1] if y!=0 else inf
            g = distance_map[x-1, y] if x!=0 else inf
            min_aller = min(d, h, g)

            if distance_map[x, y] > (min_aller + adj):
                if min_aller == d:
                    distance_map[x, y] = d+hyp
                else:
                    distance_map[x, y] = min_aller + adj

    # Retour
    for y in range(Y-1, -1, -1): # jusqu'à -1 nn inclus (donc 0), par steps de -1
        for x in range(X-1, -1, -1):
            d = distance_map[x+1, y+1] if x!=X-1 and y!=27 else inf
            b = distance_map[x, y+1] if y!=Y-1 else inf
            dr = distance_map[x+1, y] if x!=X-1 else inf
            min_retour = min(d, b, dr)
            
            if distance_map[x, y] > (min_retour + adj):
                if min_retour == d:
                    distance_map[x, y] = d+hyp
                else:
                    distance_map[x, y] = min_retour + adj

    
    return distance_map


def compute_distance(imgs, d_map):
    """
    Compute the distances for each image with respect to the reference pattern using the precomputed
    distance map. The final distance is the average of all distances from the image's contour points
    to the reference pattern.

    Args
    ----
    imgs: np.ndarray (N, 28, 28)
        Source images
    d_map: np.ndarray (28, 28)
        The precomputed distance map where each entry is the distance to the closest pattern contour
        (shortest distance to pattern)

    Return
    ------
    dist: np.ndarray (N, )
        Averaged distance to pattern for each input image.
    """

    # Default values
    dist = np.zeros(len(imgs))

    # ------------------
    # Your code here ...
    # ------------------
    
    mask = (imgs >= 1e-4).astype(np.float64)

    for n in range(len(imgs)):
        mask[n] *= d_map
        dist[n] = np.sum(mask)
        
    return dist



### 3.2 : Others
def fill_holes(img):
    mask = (img >= 1e-4).astype(np.uint8)
    
    flood = np.zeros_like(mask)
    stack = [(0, 0)]
    
    while stack:
        x, y = stack.pop()
        if x < 0 or x >= 28 or y < 0 or y >= 28:
            continue
        if flood[x, y] == 1 or mask[x, y] == 1:
            continue
        flood[x, y] = 1
        stack.extend([(x+1, y), (x-1, y), (x, y+1), (x, y-1)])
    
    filled = np.where(flood == 0, 1, mask)
    return filled

def compute_features(imgs: np.ndarray):
    """
    Compute compacity for each input image.

    Args
    ----
    imgs: np.ndarray (N, 28, 28)
        Source images

    Return
    ------
    f_peri: np.ndarray (N,)
        Estimated perimeter length for each image
    f_area: np.ndarray (N,)
        Estimated area for each image
    f_comp: np.ndarray (N,)
        Estimated compacity for each image
    f_rect: np.ndarray (N,)
        Estimated rectangularity for each image
    """

    f_peri = np.zeros(len(imgs))
    f_area = np.zeros(len(imgs))
    f_comp = np.zeros(len(imgs))
    f_rect = np.zeros(len(imgs))
    z = np.zeros(len(imgs)) 
    


    for n in range(len(imgs)):
        props = regionprops(fill_holes(imgs[n]))[0]
        f_area[n] = props.area
        f_peri[n] = props.perimeter
        f_comp[n] = f_peri[n]*f_peri[n] / f_area[n]
        f_rect[n] = f_area[n] / props.area_bbox
        z[n] = f_area[n] / f_peri[n]
    return f_peri, f_area, f_comp, f_rect#, z   #"z" must be returned only if we want to run differentiate_0_5  
                                                #"z" must NOT be returned if we want to run test_3_2