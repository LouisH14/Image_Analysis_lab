import numpy as np
import torch
from lab3 import MahalanobisClassifier
from core import image, IDX


# =============================================================================
# FAMILY 3: CLASSIFICATION & PRESENCE DETECTION (ML)
# =============================================================================

presence_model = None # Global model instance
TH_EMPTY = 10

def determine_active_player(im_obj):
    """
    Identifies which player is active based on card presence in their segment.
    """
    presence = search_present(im_obj)
    if True in presence:
        return presence.index(True) + 1
    return 0


def TEST_determine_active_player():
    for j in range(81):
        im = image(j)
        guess = determine_active_player(im)
        answer = im.show_info(IDX.ACTIVE_PLAYER)
        # Simple check: see if the guessed player number is in the ground truth string
        if str(guess) not in str(answer):
            print(f"image no: {j} guess: {guess}, answer: {answer}")
    print("test active player finito")


def extract_presence_features(segment_np):
    """
    Extracts a feature vector from a segmented area.
    Features: [Mean_R, Mean_G, Mean_B, Std_R, Std_G, Std_B]
    """
    # segment_np is (N, 3)
    mean = np.mean(segment_np, axis=0)
    std = np.std(segment_np, axis=0)
    return np.concatenate([mean, std])

def train_presence_classifier(num_train_images=41):
    """
    Trains a MahalanobisClassifier to distinguish between EMPTY and CARD PRESENT.
    """
    global presence_model
    print(f"Training presence classifier on {num_train_images} images...")
    
    features = []
    labels = []
    
    for i in range(num_train_images):
        im = image(i)
        for p in range(1, 5):
            seg = im.segment(p)
            feat = extract_presence_features(seg)
            
            # Label 0 for EMPTY, 1 for CARD PRESENT
            info = im.show_info(IDX.PLAYER_1_CARDS + (p-1))
            label = 0 if info == "EMPTY" else 1
            
            features.append(feat)
            labels.append(label)
            
    # Convert to Tensors for MahalanobisClassifier
    train_x = torch.tensor(np.array(features), dtype=torch.float32)
    train_y = torch.tensor(np.array(labels), dtype=torch.long)
    
    presence_model = MahalanobisClassifier()
    presence_model.fit(train_x, train_y)
    print("Training complete.")

def search_present(im):
    global presence_model
    players = [False, False, False, False]

    for i in range(4):
        seg = im.segment(i+1)
        
        if presence_model is not None:
            # ML approach
            feat = extract_presence_features(seg)
            feat_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
            preds, _ = presence_model.predict(feat_tensor)
            players[i] = bool(preds[0].item() == 1)
        else:
            # Heuristic approach (fallback)
            if np.std(seg) > TH_EMPTY:
                players[i] = True 
            else:
                players[i] = False
    
    return players


def TEST_search_present():

    for j in range(41):
        im = image(j)
        players = search_present(im)

        for i in range(4):
            guess = players[i]
            answer = im.show_info(IDX.PLAYER_1_CARDS+i)
            if (not verify_guess(guess, answer)):
                print(f"image no: {j} guess: {guess}, answer: {answer}")
    print("test finito")



def verify_guess(guess, answer):
    result = True

    if (guess==True) and (answer == "EMPTY"):
        result = False
    if (guess==False) and (answer != "EMPTY"):
        result = False

    return result