# =============================================================================
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from features import apply_orb

# =============================================================================
# FAMILY 4: TEMPLATE MANAGEMENT & DATABASE
# =============================================================================

def build_card_feature_db(templates_path: str):
    """
    Processes a folder containing individual card images (templates) 
    and extracts ORB descriptors for each.
    """
    folder = Path(templates_path)
    if not folder.exists():
        print(f"Error: Folder {templates_path} not found.")
        return {}

    valid_exts = [".jpg", ".jpeg", ".png"]
    template_files = [f for f in folder.glob("*") if f.suffix.lower() in valid_exts]
    
    feature_db = {}
    print(f"Building feature database from {len(template_files)} templates...")

    for file_path in tqdm(template_files):
        img = cv2.imread(str(file_path))
        if img is None:
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        kps, descriptors = apply_orb(img)
        
        if descriptors is not None:
            card_name = file_path.stem # we take directly the file's name (b_9 for b_9.png)
            feature_db[card_name] = {"descriptors": descriptors, "keypoints": kps}

    print(f"Database built with {len(feature_db)} cards.")
    return feature_db

def identify_card(query_descriptors, query_kps, feature_db, hamming_th=40, minimum_matches_th = 10):
    """
    Compares query descriptors from a table area against the database.
    Uses RANSAC to verify the geometric consistency of the matches.
    """
    if query_descriptors is None or not feature_db:
        return ["UNKNOWN"]

    # ORB uses Hamming distance. crossCheck ensures mutual best matches.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    candidates = []

    for card_name, data in feature_db.items():
        # Defensive check: ensure data is in the new dictionary format
        if isinstance(data, dict):
            db_descriptors = data["descriptors"]
            db_kps = data["keypoints"]
        else:
            # Fallback for old database format (no RANSAC possible)
            db_descriptors = data
            db_kps = None

        # Find matches
        matches = bf.match(query_descriptors, db_descriptors)
        
        # Filter by quality
        good_matches = [m for m in matches if m.distance < hamming_th]
        
        # Geometric verification requires at least 4 matches to calculate perspective
        if db_kps is not None and query_kps is not None and len(good_matches) >= max(4, minimum_matches_th):
            src_pts = np.float32([db_kps[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([query_kps[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # RANSAC filters out "outliers" (matches that don't fit the card's plane)
            _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if mask is not None:
                # We count only the inliers (mathematically consistent matches)
                num_inliers = np.sum(mask)
                if num_inliers >= minimum_matches_th:
                    candidates.append((card_name, int(num_inliers)))
        elif len(good_matches) >= minimum_matches_th:
            # Fallback count if RANSAC is not possible
            candidates.append((card_name, len(good_matches)))

    # Sort candidates by number of good matches (highest first)
    candidates.sort(key=lambda x: x[1], reverse=True)   

    if not candidates:
        return ["EMPTY"]
        
    return [f"{c[0]} ({c[1]})" for c in candidates]

def predict_table_state(im_obj, feature_db, presence_model=None, max_number_values = 6, match_threshold = 20):
    """
    Analyzes a full table image and identifies cards for all players and the center.
    
    Args:
        im_obj: The 'image' object from core.py.
        feature_db: The dictionary of card descriptors.
        presence_model: The trained MahalanobisClassifier (optional).
        max_number_values: Number of top candidates to return.
        match_threshold: Minimum number of ORB matches required to accept a card identification.
    """
    from presence import search_present

    # 1. Determine which areas actually contain cards
    # This prevents matching ORB against a plain tablecloth (which produces noise)
    presence_mask = search_present(im_obj, model=presence_model)
    
    results = {}
    positions = ["Center", "Player 1", "Player 2", "Player 3", "Player 4"]

    for i, is_present in enumerate(presence_mask):

        # We keep the search active for all areas as presence detection 
        # can be bypassed by the ORB match threshold itself.

        # 2. Extract 2D area, keypoints and descriptors
        area = im_obj.area(i)
        kps, descriptors = apply_orb(area)
        
        # 3. Match against the database
        card_list = identify_card(descriptors, kps, feature_db, minimum_matches_th=match_threshold)
        results[positions[i]] = card_list[:max_number_values]
            
    return results
