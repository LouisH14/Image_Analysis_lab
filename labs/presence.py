import numpy as np
import torch
from lab3 import MahalanobisClassifier
from sklearn.metrics import accuracy_score, f1_score
from core import image, IDX


# =============================================================================
# FAMILY 3: CLASSIFICATION & PRESENCE DETECTION (ML)
# =============================================================================

presence_model = None # Global model instance
TH_EMPTY = 10

def determine_active_player(im_obj, model=None):
    """
    Identifies which player is active based on card presence in their segment.
    """
    presence = search_present(im_obj, model=model)
    player_presence = presence[1:] # Exclude the center card (index 0)
    if True in player_presence:
        return player_presence.index(True) + 1
    return 0


def TEST_determine_active_player(model=None):
    for j in range(81):
        im = image(j)
        guess = determine_active_player(im, model=model)
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
    # Reshape the rectangular patch (H, W, 3) to a flat list of pixels (N, 3)
    pixels = segment_np.reshape(-1, 3)
    mean = np.mean(pixels, axis=0)
    std = np.std(pixels, axis=0)
    return np.concatenate([mean, std])

def train_presence_classifier(num_train_images=10):
    """
    Trains a MahalanobisClassifier to distinguish between EMPTY and CARD PRESENT.
    """
    global presence_model
    print(f"Training presence classifier on {num_train_images} images...")
    
    features = []
    labels = []
    
    for i in range(num_train_images):
        im = image(i)
        for p in range(5): # 0 (center) to 4 (players)
            seg = im.segment(p)
            feat = extract_presence_features(seg)
            
            # Label 0 for EMPTY, 1 for CARD PRESENT
            if p == 0:
                info = im.show_info(IDX.CENTER_CARD)
            else:
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
    return presence_model

def search_present(im, model=None):
    global presence_model
    current_model = model if model is not None else presence_model
    presence_results = [False] * 5 # [Center, P1, P2, P3, P4]

    for i in range(5):
        seg = im.segment(i)
        
        if current_model is not None:
            # ML approach
            feat = extract_presence_features(seg)
            feat_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
            preds, _ = current_model.predict(feat_tensor)
            presence_results[i] = bool(preds[0].item() == 1)
        else:
            # Heuristic approach (fallback)
            if np.std(seg) > TH_EMPTY:
                presence_results[i] = True 
            else:
                presence_results[i] = False
    
    return presence_results


def TEST_search_present(model=None, num_train_images=41, num_total_images=81):
    """
    Tests the search_present function against ground truth for a range of images
    and calculates overall accuracy and F1-score.
    """
    global presence_model
    current_model = model if model is not None else presence_model
    if current_model is None:
        print("Presence classifier not trained. Please train it first.")
        return 0
    
    all_guesses = []
    all_answers = []
    
    print(f"\nTesting presence detection on images from {num_train_images} to {num_total_images-1}...")
    
    for j in range(num_train_images, num_total_images): # Test on images not used for training
        im = image(j)
        presence_results = search_present(im, model=current_model)

        for i in range(5): # Iterate through all 5 areas (Center + P1 to P4)
            guess_bool = presence_results[i]
            
            if i == 0: # Center Card
                answer_str = im.show_info(IDX.CENTER_CARD)
            else: # Players 1-4
                answer_str = im.show_info(IDX.PLAYER_1_CARDS + (i-1))
            
            # Convert ground truth string to boolean for comparison
            answer_bool = (answer_str != "EMPTY")
            
            all_guesses.append(guess_bool)
            all_answers.append(answer_bool)

            # Report individual discrepancies
            if guess_bool != answer_bool:
                print(f"Image {j}, Area {i}: GUESS={guess_bool} (predicted), ANSWER={answer_str} (ground truth)")

    # Calculate overall metrics
    accuracy = accuracy_score(all_answers, all_guesses)
    f1 = f1_score(all_answers, all_guesses)
    
    print(f"\n--- Presence Detection Test Results ---")
    print(f"Total Segments Tested: {len(all_guesses)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("---------------------------------------")
    print("Presence detection test finished.")


def verify_guess(guess, answer): # This function can be simplified or removed as the new TEST_search_present directly compares booleans
    """
    Helper to verify a single guess against its answer.
    (Note: The new TEST_search_present performs direct boolean comparison.)
    """
    result = True

    # The guess is a boolean (True/False for presence)
    # The answer is a string ("EMPTY" or card name)
    
    if (guess == True) and (answer == "EMPTY"):
        result = False
    if (guess == False) and (answer != "EMPTY"):
        result = False

    return result
