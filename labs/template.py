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
        _, descriptors = apply_orb(img)
        
        if descriptors is not None:
            card_name = file_path.stem # we take directly the file's name (b_9 for b_9.png)
            feature_db[card_name] = descriptors

    print(f"Database built with {len(feature_db)} cards.")
    return feature_db

def identify_card(query_descriptors, feature_db, hamming_th=40, minimum_matches_th = 10):
    """
    Compares query descriptors from a table segment against the database.
    Returns a list of card names that meet the match threshold.
    """
    if query_descriptors is None or not feature_db:
        return ["UNKNOWN"]

    # ORB uses Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    candidates = []

    for card_name, db_descriptors in feature_db.items():
        # Match descriptors
        potential_matches = bf.match(query_descriptors, db_descriptors)
        
        # Sort potential_matches by distance (lower is better)
        potential_matches = sorted(potential_matches, key=lambda x: x.distance)
        
        # among all potential matches, which ones are the good matches
        good_matches = [m for m in potential_matches if m.distance < hamming_th] # hamming_th found iteratively
        
        num_good = len(good_matches)
        
        # Accept any card that meets the minimum match threshold
        if num_good >= minimum_matches_th: # minimum_matches_th found iteratively too
            candidates.append((card_name, num_good))

    # Sort candidates by number of good matches (highest first)
    candidates.sort(key=lambda x: x[1], reverse=True)   

    if not candidates:
        return ["EMPTY"]
        
    return [f"{c[0]} ({c[1]})" for c in candidates]

def predict_table_state(im_obj, feature_db, presence_model=None, max_number_values = 6):
    """
    Analyzes a full table image and identifies cards for all players and the center.
    
    Args:
        im_obj: The 'image' object from core.py.
        feature_db: The dictionary of card descriptors.
        presence_model: The trained MahalanobisClassifier (optional).
    """
    from presence import search_present

    # 1. Determine which areas actually contain cards
    # This prevents matching ORB against a plain tablecloth (which produces noise)
    presence_mask = search_present(im_obj, model=presence_model)
    
    results = {}
    positions = ["Center", "Player 1", "Player 2", "Player 3", "Player 4"]

    for i, is_present in enumerate(presence_mask):

        """if not is_present:
            results[positions[i]] = "EMPTY"
        else: ==> the presence detection wasn't working good"""
        
        # 2. Extract 2D segment and descriptors
        seg = im_obj.segment(i)
        _, descriptors = apply_orb(seg)
        
        # 3. Match against the database
        card_list = identify_card(descriptors, feature_db)
        results[positions[i]] = card_list[:max_number_values]
            
    return results
