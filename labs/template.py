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

def identify_card(query_descriptors, query_kps, feature_db, hamming_th=40, minimum_matches_th=10, max_iterations=5):
    """
    Compares query descriptors from a table area against the database.
    Uses Iterative RANSAC (Method 1) to identify multiple cards by isolating their features.
    """
    if query_descriptors is None or len(query_descriptors) < minimum_matches_th or not feature_db:
        return ["UNKNOWN"]

    # Force descriptors to be a numpy array to support boolean indexing/masking
    curr_descriptors = np.array(query_descriptors)
    curr_kps = list(query_kps)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    found_cards = []

    for _ in range(max_iterations):
        if len(curr_descriptors) < minimum_matches_th:
            break

        best_candidate = None
        best_inlier_count = 0
        best_inlier_indices = []

        for card_name, data in feature_db.items():
            # Support both new dictionary format and legacy array format
            if isinstance(data, dict):
                db_descriptors = data.get("descriptors")
                db_kps = data.get("keypoints")
            else:
                db_descriptors = data
                db_kps = None

            if db_descriptors is None or len(db_descriptors) == 0:
                continue

            matches = bf.match(curr_descriptors, db_descriptors)
            good_matches = [m for m in matches if m.distance < hamming_th]
            
            # Verification using RANSAC (only if KeyPoints are available for both)
            if db_kps is not None and len(good_matches) >= max(4, minimum_matches_th):
                src_pts = np.float32([db_kps[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([curr_kps[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if mask is not None:
                    num_inliers = int(np.sum(mask))
                    if num_inliers > best_inlier_count:
                        best_inlier_count = num_inliers
                        best_candidate = card_name
                        best_inlier_indices = [good_matches[i].queryIdx for i, val in enumerate(mask.ravel()) if val]
            
            # Fallback if RANSAC is not possible (no KeyPoints in DB)
            elif db_kps is None and len(good_matches) >= minimum_matches_th:
                if len(good_matches) > best_inlier_count:
                    best_inlier_count = len(good_matches)
                    best_candidate = card_name
                    best_inlier_indices = [m.queryIdx for m in good_matches]

        if best_candidate and best_inlier_count >= minimum_matches_th:
            found_cards.append(f"{best_candidate} ({best_inlier_count})")
            
            remaining_mask = np.ones(len(curr_descriptors), dtype=bool)
            # Mark the query indices used by the best match for removal
            for idx in best_inlier_indices:
                if idx < len(remaining_mask):
                    remaining_mask[idx] = False
            
            curr_descriptors = curr_descriptors[remaining_mask]
            curr_kps = [curr_kps[i] for i, is_val in enumerate(remaining_mask) if is_val]
        else:
            break

    if not found_cards:
        return ["EMPTY"]

    return found_cards

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
