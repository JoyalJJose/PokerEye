import cv2 as cv
import numpy as np
from treys import Card, Evaluator
import random
import os

# Global variables for game state
board = []
hand1 = []
hand2 = []
game_stage_counter = 0
numHands = 0

# Store detected images for display
hand1_detected = None
hand2_detected = None
board_detected = None

def rearrange_points(pts):
    """Reorder 4 points to top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def resize_with_padding(img, target_width, target_height):
    """
    Resize image to target size while maintaining aspect ratio.
    Adds padding (black background) to fill to target size without cropping.
    """
    if img is None:
        return None
    
    h, w = img.shape[:2]
    
    # Calculate scaling factor to fit within target size
    scale = min(target_width / w, target_height / h)
    
    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_AREA)
    
    # Create target-sized image with black background
    result = np.zeros((target_height, target_width), dtype=img.dtype)
    
    # Calculate padding to center the image
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    
    # Place resized image in center
    result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return result

def load_reference_images(folder_path, image_type="rank"):
    """
    Load reference images from the specified folder.
    Returns a dictionary mapping image names (without extension) to image arrays.
    Handles naming patterns like "rank_A.png", "A.png", "suit_spades.png", "spades.png"
    """
    reference_images = {}
    
    if not os.path.exists(folder_path):
        print(f"Warning: Reference folder '{folder_path}' does not exist.")
        return reference_images
    
    # Look for images that match the type (rank or suit)
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            name_without_ext = os.path.splitext(filename)[0]
            
            # Check if filename starts with the image type (e.g., "rank_A.png" or "suit_spades.png")
            if name_without_ext.startswith(f"{image_type}_"):
                # Extract the actual name (e.g., "A" from "rank_A.png")
                actual_name = name_without_ext[len(f"{image_type}_"):]
            elif name_without_ext == image_type:
                # Handle case where filename is just "rank.png" or "suit.png"
                actual_name = name_without_ext
            else:
                # Skip files that don't match the type
                continue
            
            img_path = os.path.join(folder_path, filename)
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            if img is not None:
                # Use uppercase for ranks, lowercase for suits
                if image_type == "rank":
                    actual_name = actual_name.upper()
                else:
                    actual_name = actual_name.lower()
                reference_images[actual_name] = img
    
    return reference_images

def find_best_match(extracted_img, reference_images):
    """
    Find the best matching reference image using absdiff.
    Returns (best_match_name, best_score) where lower score is better.
    """
    if extracted_img is None or len(reference_images) == 0:
        return None, float('inf')
    
    best_match = None
    best_score = float('inf')
    
    for name, ref_img in reference_images.items():
        # Ensure both images are the same size
        if extracted_img.shape != ref_img.shape:
            # Resize reference image to match extracted image
            ref_img = cv.resize(ref_img, (extracted_img.shape[1], extracted_img.shape[0]), interpolation=cv.INTER_AREA)
        
        # Calculate absolute difference
        diff = cv.absdiff(extracted_img, ref_img)
        
        # Calculate difference score: sum of all pixel differences divided by 255
        score = np.sum(diff) / 255.0
        
        if score < best_score:
            best_score = score
            best_match = name
    
    return best_match, best_score

def identify_rank_suit(rank_cropped, suit_cropped, rank_references, suit_references):
    """
    Identifies the rank and suit from cropped images using template matching.
    Returns (identified_rank, identified_suit) in treys format.
    """
    identified_rank = None
    identified_suit = None
    
    # Identify rank
    if rank_cropped is not None and rank_references is not None and len(rank_references) > 0:
        rank_match, _ = find_best_match(rank_cropped, rank_references)
        identified_rank = rank_match.upper() if rank_match else None
    
    # Identify suit
    if suit_cropped is not None and suit_references is not None and len(suit_references) > 0:
        suit_match, _ = find_best_match(suit_cropped, suit_references)
        # Convert suit to treys format (hearts -> h, diamonds -> d, clubs -> c, spades -> s)
        if suit_match:
            suit_lower = suit_match.lower()
            suit_map = {"hearts": "h", "diamonds": "d", "clubs": "c", "spades": "s"}
            identified_suit = suit_map.get(suit_lower, suit_lower[0] if suit_lower else None)
    
    return identified_rank, identified_suit

def process_card(card, rank_references=None, suit_references=None):
    # Standard sizes for rank and suit images
    RANK_WIDTH = 100
    RANK_HEIGHT = 150
    SUIT_WIDTH = 80
    SUIT_HEIGHT = 100
    
    height, width = card.shape[:2]
    # Use percentages to calculate the ROI for the top left corner
    x_start = int(width * 0.03)
    x_end = int(width * 0.15)
    y_start = int(height * 0.025)
    y_end = int(height * 0.275)
    top_left_region = card[y_start:y_end, x_start:x_end]  # take the top left corner using percentages

    # Maintain aspect ratio while resizing
    desired_zoom_height = 300
    h, w = top_left_region.shape[:2]
    aspect_ratio = w / h
    new_width = int(desired_zoom_height * aspect_ratio)
    zoomed_display = cv.resize(top_left_region, (new_width, desired_zoom_height)) # maintain aspect ratio
    # cv.imshow(f"Top-Left Zoom {idx}", zoomed_display)
    
    # HSV based thresholding
    imghsv = cv.cvtColor(zoomed_display, cv.COLOR_BGR2HSV)
    lower = np.array([0, 0, 150])
    upper = np.array([179, 80, 255]) 
    mask_hsv = cv.inRange(imghsv, lower, upper)#produces a binary mask where pixels within the range become 255 (white), others become 0 (black).
    
    # Otsu  thresholding on grayscale
    gray = cv.cvtColor(zoomed_display, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray) #Equlise the gray image 
    _, mask_gray = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    # Combine both masks
    mask = cv.bitwise_or(mask_hsv, mask_gray)#Merge the 2 mask togother 
    #So this mask allows us to cpature white or bright areas 
    # Clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    # get the inverse mask
    mask = cv.bitwise_not(mask)
    
    # cv.imshow(f"White Mask {idx}", mask) #sHOW MASK 
    contour_mask, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) # fIND THE CONOTURS BASED ON THE  BINARY MASK
    
    # Separate rank and suit by analyzing contours
    rank_cropped = None
    suit_cropped = None
    
    if len(contour_mask) > 0:
        # Filter contours by area to remove noise
        min_area = 50  # Minimum area threshold
        valid_contours = [c for c in contour_mask if cv.contourArea(c) >= min_area]
        
        if len(valid_contours) > 0:
            # Get bounding boxes for all valid contours
            bounding_boxes = []
            for contour in valid_contours:
                x, y, w, h = cv.boundingRect(contour)
                bounding_boxes.append((x, y, w, h, contour))
            
            # Sort by y-position (top to bottom)
            bounding_boxes.sort(key=lambda b: b[1])
            
            # Split into rank (upper) and suit (lower) based on vertical position
            mask_height = mask.shape[0]
            midpoint_y = mask_height // 2
            
            rank_boxes = [b for b in bounding_boxes if b[1] + b[3]//2 < midpoint_y]
            suit_boxes = [b for b in bounding_boxes if b[1] >= midpoint_y]
            
            # If no clear split, use first half for rank, second half for suit
            if len(rank_boxes) == 0 or len(suit_boxes) == 0:
                mid_idx = len(bounding_boxes) // 2
                rank_boxes = bounding_boxes[:mid_idx] if mid_idx > 0 else []
                suit_boxes = bounding_boxes[mid_idx:] if mid_idx < len(bounding_boxes) else []
            
            # Find combined bounding box for rank
            if len(rank_boxes) > 0:
                min_x = min(b[0] for b in rank_boxes)
                min_y = min(b[1] for b in rank_boxes)
                max_x = max(b[0] + b[2] for b in rank_boxes)
                max_y = max(b[1] + b[3] for b in rank_boxes)
                
                # Crop with minimal padding (1 pixel to ensure we capture edges)
                padding = 1
                min_x = max(0, min_x - padding)
                min_y = max(0, min_y - padding)
                max_x = min(mask.shape[1], max_x + padding)
                max_y = min(mask.shape[0], max_y + padding)
                
                rank_cropped = mask[min_y:max_y, min_x:max_x]
                
                # Resize rank to standard size while maintaining aspect ratio
                rank_cropped = resize_with_padding(rank_cropped, RANK_WIDTH, RANK_HEIGHT)
            
            # Find combined bounding box for suit
            if len(suit_boxes) > 0:
                min_x = min(b[0] for b in suit_boxes)
                min_y = min(b[1] for b in suit_boxes)
                max_x = max(b[0] + b[2] for b in suit_boxes)
                max_y = max(b[1] + b[3] for b in suit_boxes)
                
                # Crop with minimal padding (1 pixel to ensure we capture edges)
                padding = 1
                min_x = max(0, min_x - padding)
                min_y = max(0, min_y - padding)
                max_x = min(mask.shape[1], max_x + padding)
                max_y = min(mask.shape[0], max_y + padding)
                
                suit_cropped = mask[min_y:max_y, min_x:max_x]
                
                # Resize suit to standard size while maintaining aspect ratio
                suit_cropped = resize_with_padding(suit_cropped, SUIT_WIDTH, SUIT_HEIGHT)
    
    # Identify rank and suit using template matching
    identified_rank, identified_suit = identify_rank_suit(rank_cropped, suit_cropped, rank_references, suit_references)
    
    return rank_cropped, suit_cropped, identified_rank, identified_suit

def calculate_odds(hands, board, num_simulations=10000):
    """
    Calculate odds (winning probabilities) for each player at any stage of the game.
    
    Args:
        hands: List of player hands, where each hand is a list of 2 Card objects
        board: List of community cards (0-5 cards)
        num_simulations: Number of random simulations (default: 10000)
    
    Returns:
        Dictionary with win percentages for each player
    """
    if not hands or len(hands) < 2:
        print("Error: Need at least 2 players to calculate odds.")
        return None
    
    if len(board) > 5:
        print("Error: Board cannot have more than 5 cards.")
        return None
    
    # Get all known cards (all player hands + board)
    known_cards = []
    for hand in hands:
        known_cards.extend(hand)
    known_cards.extend(board)
    
    # Create full deck (52 cards)
    full_deck = []
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    suits = ['h', 'd', 'c', 's']
    for rank in ranks:
        for suit in suits:
            full_deck.append(Card.new(rank + suit))
    
    # Remove known cards from deck
    remaining_deck = [card for card in full_deck if card not in known_cards]
    
    num_players = len(hands)
    wins = [0] * num_players
    ties = [0] * num_players
    evaluator = Evaluator()
    
    # If board is complete (5 cards), evaluate directly (exact result)
    if len(board) == 5:
        hand_scores = []
        for hand in hands:
            score = evaluator.evaluate(board, hand)
            hand_scores.append(score)
        
        best_score = min(hand_scores)
        winners = [i for i, score in enumerate(hand_scores) if score == best_score]
        
        if len(winners) == 1:
            wins[winners[0]] = 1
        else:
            for winner in winners:
                ties[winner] = 1
        
        total_scenarios = 1
    else:
        # Need to simulate remaining cards
        cards_needed = 5 - len(board)
        if len(remaining_deck) < cards_needed:
            print("Error: Not enough cards remaining in deck.")
            return None
        
        random.seed(42)
        
        for i in range(num_simulations):
            # Complete the board with random cards
            additional_cards = random.sample(remaining_deck, cards_needed)
            complete_board = board + additional_cards
            
            # Evaluate each hand
            hand_scores = []
            for hand in hands:
                score = evaluator.evaluate(complete_board, hand)
                hand_scores.append(score)
            
            # Lower score is better in treys
            best_score = min(hand_scores)
            winners = [i for i, score in enumerate(hand_scores) if score == best_score]
            
            if len(winners) == 1:
                wins[winners[0]] += 1
            else:
                for winner in winners:
                    ties[winner] += 1
        
        total_scenarios = num_simulations
    
    # Calculate percentages
    results = {}
    for i in range(num_players):
        win_pct = (wins[i] / total_scenarios) * 100
        tie_pct = (ties[i] / total_scenarios) * 100
        results[i] = {
            'wins': wins[i],
            'ties': ties[i],
            'win_percentage': win_pct,
            'tie_percentage': tie_pct
        }
    
    return results

def pre_process(img):
    """
    Pre-processes the image to detect and extract card regions.
    Returns (captured_cards, card_contours) where:
    - captured_cards: list of transformed card images
    - card_contours: list of contours for drawing bounding boxes
    """
    imgcon = img.copy()
    
    # Convert to grayscale and equalize histogram
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_img = cv.equalizeHist(gray_img)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv.GaussianBlur(gray_img, (5, 5), 0)
    
    # Canny edge detection with adjusted thresholds
    median = np.median(blurred)
    sigma = 0.33
    lower_threshold = int(max(0, (1.0 - sigma) * median))
    upper_threshold = int(min(255, (1.0 + sigma) * median))
    cannyedge = cv.Canny(blurred, lower_threshold, upper_threshold)
    
    # Find contours
    contours, _ = cv.findContours(cannyedge, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    
    captured_cards = []
    card_contours = []
    width, height = 400, 600
    p2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    
    print(f"\nTotal contours found: {len(contours)}")
    print("=" * 60)
    card_count = 0
    for i, contour in enumerate(contours):
        area = cv.contourArea(contour)
        if i < 20:
            print(f"Contour {i}: Area = {area:.0f}")
        if area < 10000:
            continue
        
        # Calculate solidity
        hull = cv.convexHull(contour)
        hull_area = cv.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / hull_area
        
        # Approximate the contour to a polygon
        perimeter = cv.arcLength(contour, True)
        epsilon = 0.02 * perimeter
        approx = cv.approxPolyDP(contour, epsilon, True)
        
        if i < 20:
            print(f"  -> Corners: {len(approx)}, Solidity: {solidity:.2f}")
        
        # Check for 4 sided shape
        if len(approx) == 4 and solidity > 0.80:
            print(f"\nCARD DETECTED (Contour {i})")
            print(f"  Area: {area:.0f}, Solidity: {solidity:.2f}")
            print(f"  Corners: {approx.reshape(4, 2)}")
            
            pts = np.float32(approx.reshape(4, 2))
            ordered_pts = rearrange_points(pts)
            
            # Apply perspective transform
            matrix = cv.getPerspectiveTransform(ordered_pts, p2)
            output = cv.warpPerspective(img, matrix, (width, height))
            captured_cards.append(output)
            card_contours.append(approx)
            
            card_count += 1
            if card_count >= 10:
                break
    
    print(f"\n{'=' * 60}")
    print(f"Total cards captured: {len(captured_cards)}")
    print("=" * 60)
    
    return captured_cards, card_contours

def display_detected_cards(img, card_contours, card_labels):
    """
    Creates an image with bounding boxes and rank/suit labels.
    Returns the image with annotations.
    """
    img_display = img.copy()
    colors = [(255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 128, 255)]
    
    # First draw all contours and rectangles
    for idx, approx in enumerate(card_contours):
        color = colors[idx % len(colors)]
        cv.drawContours(img_display, [approx], -1, color, 4)
        
        x, y, w, h = cv.boundingRect(approx)
        cv.rectangle(img_display, (x, y), (x + w, y + h), (0, 255, 0), 3)
    
    # Then draw all labels last so they appear on top
    for idx, (approx, label) in enumerate(zip(card_contours, card_labels)):
        x, y, w, h = cv.boundingRect(approx)
        cv.putText(img_display, label, (x, y - 10), 
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Red color for label
    
    return img_display

# Load reference images for template matching
reference_folder = "rankSuitImages"
rank_references = load_reference_images(reference_folder, image_type="rank")
suit_references = load_reference_images(reference_folder, image_type="suit")

while True:
    print("--------------------------------")
    print("Welcome to the PokerEye")
    print("--------------------------------")
    print("Please select an option:")
    print("1. Insert image of player hand")
    print("2. Insert image of community cards")
    print("3. Calculate odds for each player")
    print("0. Exit")

    # Take user option input
    while True:
        option = input("Enter your option: ")
        try:
            option = int(option)
            if option in [0, 1, 2, 3]:
                break
            else:
                print("Invalid option. Please enter 0, 1, 2, or 3.")
        except ValueError:
            print("Invalid input. Please enter a number (0, 1, 2, or 3).")

    # Exit the program
    if option == 0:
        print("--------------------------------")
        print("Exiting.....")
        print("--------------------------------")
        break

    # Read poker hand image and process it
    elif option == 1:

        hand1_img = cv.imread("Project/projectImages/hand1.jpg") 
        hand1_img = cv.resize(hand1_img, (hand1_img.shape[1] // 8, hand1_img.shape[0] // 8))

        # Pre-process image to extract card regions
        captured_cards, card_contours = pre_process(hand1_img)

        # Process each captured card to isolate and identify rank/suit
        detected_cards = []
        card_labels = []
        hand1 = []
        for card in captured_cards:
            rank_cropped, suit_cropped, identified_rank, identified_suit = process_card(
                card, rank_references=rank_references, suit_references=suit_references)
            
            # Store detected card in treys format if both rank and suit are identified
            if identified_rank and identified_suit:
                card_string = f"{identified_rank}{identified_suit}"
                detected_cards.append(f"Card.new('{card_string}')")
                card_labels.append(card_string)
                # Add Card object to hand1
                hand1.append(Card.new(card_string))
            else:
                card_labels.append("Unknown")

        # Create the image with bounding boxes and labels
        hand1_detected = display_detected_cards(hand1_img, card_contours, card_labels)
        

        hand2_img = cv.imread("Project/projectImages/hand2.jpg") 
        hand2_img = cv.resize(hand2_img, (hand2_img.shape[1] // 8, hand2_img.shape[0] // 8))

        # Pre-process image to extract card regions
        captured_cards, card_contours = pre_process(hand2_img)

        # Process each captured card to isolate and identify rank/suit
        detected_cards = []
        card_labels = []
        hand2 = []
        for card in captured_cards:
            rank_cropped, suit_cropped, identified_rank, identified_suit = process_card(
                card, rank_references=rank_references, suit_references=suit_references)
            
            # Store detected card in treys format if both rank and suit are identified
            if identified_rank and identified_suit:
                card_string = f"{identified_rank}{identified_suit}"
                detected_cards.append(f"Card.new('{card_string}')")
                card_labels.append(card_string)
                # Add Card object to hand2
                hand2.append(Card.new(card_string))
            else:
                card_labels.append("Unknown")

        # Create the image with bounding boxes and labels
        hand2_detected = display_detected_cards(hand2_img, card_contours, card_labels)

    # Read community cards image and process it
    elif option == 2:
        board = []

        game_stage_counter += 1
        if game_stage_counter == 1:
            board_img = cv.imread("Project/projectImages/flop.jpg")
        elif game_stage_counter == 2:
            board_img = cv.imread("Project/projectImages/turn.jpg")
        elif game_stage_counter > 2:
            board_img = cv.imread("Project/projectImages/river.jpg")

        board_img = cv.resize(board_img, (board_img.shape[1] // 8, board_img.shape[0] // 8))

        # Pre-process image to extract card regions
        captured_cards, card_contours = pre_process(board_img)

        # Process each captured card to isolate and identify rank/suit
        detected_cards = []
        card_labels = []
        board = []
        for card in captured_cards:
            rank_cropped, suit_cropped, identified_rank, identified_suit = process_card(
                card, rank_references=rank_references, suit_references=suit_references)
            
            # Store detected card in treys format if both rank and suit are identified
            if identified_rank and identified_suit:
                card_string = f"{identified_rank}{identified_suit}"
                detected_cards.append(f"Card.new('{card_string}')")
                card_labels.append(card_string)
                # Add Card object to board
                board.append(Card.new(card_string))
            else:
                card_labels.append("Unknown")

        # Create the image with bounding boxes and labels
        board_detected = display_detected_cards(board_img, card_contours, card_labels)

    # Calculate odds for each player
    elif option == 3:
        player_hands = [hand1, hand2]
        
        # Calculate odds using current global values
        results = calculate_odds(player_hands, board, num_simulations=10000)
        
        if results:
            print()
            # Print player hands with win percentages
            for player_idx, result in results.items():
                hand = player_hands[player_idx]
                hand_str = Card.ints_to_pretty_str(hand)
                print(f"Player {player_idx + 1}:{hand_str} {result['win_percentage']:.1f}%")
            
            # Print board
            if len(board) > 0:
                board_str = Card.ints_to_pretty_str(board)
                print(f"Board:{board_str}")
            print()
            
            # Display all images with win percentages

             # Display Board (no win percentage for board)
            if board_detected is not None:
                board_display = board_detected.copy()
                cv.imshow("Board", board_display)
            
            # Display Hand 1 with win percentage
            if hand1_detected is not None:
                hand1_display = hand1_detected.copy()
                win_pct = results[0]['win_percentage']
                label_text = f"Win: {win_pct:.1f}%"
                cv.putText(hand1_display, label_text, (10, 40), 
                          cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)  # Blue color
                cv.imshow("Hand 1", hand1_display)
            
            # Display Hand 2 with win percentage
            if hand2_detected is not None:
                hand2_display = hand2_detected.copy()
                win_pct = results[1]['win_percentage']
                label_text = f"Win: {win_pct:.1f}%"
                cv.putText(hand2_display, label_text, (10, 40), 
                          cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)  # Blue color
                cv.imshow("Hand 2", hand2_display)
            
            cv.waitKey(0)
            cv.destroyAllWindows()

