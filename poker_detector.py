import os
import cv2
import cvzone
import math
import time
from ultralytics import YOLO

# Import configuration
import config

class PokerHandDetector:
    """Class to analyze and determine poker hands based on detected cards"""
    
    def findPokerHand(self, hand):
        """Determine the poker hand based on the detected cards"""
        ranks = []
        suits = []
        possible_ranks = []

        for card in hand:
            if len(card) == 2:
                rank = card[0]
                suit = card[1]
            else:
                rank = card[0:2]
                suit = card[2]
                
            if rank == "A":
                rank = 14
            elif rank == "K":
                rank = 13
            elif rank == "Q":
                rank = 12
            elif rank == "J":
                rank = 11
            ranks.append(int(rank))
            suits.append(suit)

        sorted_ranks = sorted(ranks)

        # Royal Flush and Straight Flush and Flush
        if suits.count(suits[0]) == 5:  # Check for Flush
            if 14 in sorted_ranks and 13 in sorted_ranks and 12 in sorted_ranks and 11 in sorted_ranks and 10 in sorted_ranks:
                possible_ranks.append(10)
            elif all(sorted_ranks[i] == sorted_ranks[i - 1] + 1 for i in range(1, len(sorted_ranks))):
                possible_ranks.append(9)
            else:
                possible_ranks.append(6)  # -- Flush

        # Straight
        if all(sorted_ranks[i] == sorted_ranks[i - 1] + 1 for i in range(1, len(sorted_ranks))):
            possible_ranks.append(5)

        hand_unique_vals = list(set(sorted_ranks))

        # Four of a kind and Full House
        if len(hand_unique_vals) == 2:
            for val in hand_unique_vals:
                if sorted_ranks.count(val) == 4:  # --- Four of a kind
                    possible_ranks.append(8)
                if sorted_ranks.count(val) == 3:  # --- Full house
                    possible_ranks.append(7)

        # Three of a Kind and Two Pair
        if len(hand_unique_vals) == 3:
            for val in hand_unique_vals:
                if sorted_ranks.count(val) == 3:  # -- three of a kind
                    possible_ranks.append(4)
            
            # Count pairs for two pair detection
            pair_count = sum(1 for val in hand_unique_vals if sorted_ranks.count(val) == 2)
            if pair_count == 2:  # -- two pair
                possible_ranks.append(3)

        # Pair
        if len(hand_unique_vals) == 4:
            possible_ranks.append(2)

        if not possible_ranks:
            possible_ranks.append(1)
            
        poker_hand_ranks = {
            10: "Royal Flush", 
            9: "Straight Flush", 
            8: "Four of a Kind", 
            7: "Full House", 
            6: "Flush",
            5: "Straight", 
            4: "Three of a Kind", 
            3: "Two Pair", 
            2: "Pair", 
            1: "High Card"
        }
        
        output = poker_hand_ranks[max(possible_ranks)]
        return output

class CardDetector:
    """Main class for real-time card detection and poker hand analysis"""
    
    def __init__(self):
        self.model_path = config.MODEL_PATH
        self.class_names = config.CLASS_NAMES
        self.conf_threshold = config.CONFIDENCE_THRESHOLD
        self.poker_detector = PokerHandDetector()
        self.fps = 0
        self.prev_time = 0
        
        # Load the YOLOv8 model
        self.load_model()
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(config.CAMERA_ID)
        self.cap.set(3, config.CAMERA_WIDTH)
        self.cap.set(4, config.CAMERA_HEIGHT)
    
    def load_model(self):
        """Load the YOLO model from the specified path"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            self.model = YOLO(self.model_path)
            print(f"Model loaded successfully: {self.model_path}")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def process_frame(self, frame):
        """Process a single frame for card detection"""
        # Calculate FPS
        current_time = time.time()
        if (current_time - self.prev_time) > 0:
            self.fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time
        
        # Run YOLOv8 inference
        results = self.model(frame, stream=True)
        
        # List to store detected cards
        detected_cards = []
        
        # Process detection results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                
                # Get confidence and class
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                # Skip if confidence is below threshold or class ID is invalid
                if conf < self.conf_threshold or cls_id >= len(self.class_names):
                    continue
                
                # Get card name
                card_name = self.class_names[cls_id]
                
                # Draw bounding box
                if config.DRAW_BOUNDING_BOXES:
                    cvzone.cornerRect(
                        frame, 
                        (x1, y1, w, h), 
                        colorC=config.COLORS['bounding_box'],
                        colorR=config.COLORS['corner_rect'],
                        rt=2
                    )
                
                # Add text with card name and confidence
                conf_text = f" {conf:.2f}" if config.SHOW_CONFIDENCE else ""
                cvzone.putTextRect(
                    frame, 
                    f'{card_name}{conf_text}', 
                    (max(0, x1), max(35, y1)), 
                    scale=1, 
                    thickness=1
                )
                
                # Add card to detected list
                detected_cards.append(card_name)
        
        # Remove duplicates
        unique_cards = list(set(detected_cards))
        
        # Draw FPS counter
        cv2.putText(
            frame, 
            f"FPS: {int(self.fps)}", 
            (20, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )
        
        # Process poker hand if 5 cards are detected
        if len(unique_cards) == 5:
            result = self.poker_detector.findPokerHand(unique_cards)
            
            # Display the poker hand
            cvzone.putTextRect(
                frame, 
                f'Your Hand: {result}', 
                (frame.shape[1]//2 - 200, 75), 
                scale=2, 
                thickness=3,
                colorR=config.COLORS['text_bg'],
                colorT=config.COLORS['text']
            )
        
        # Display detected cards count
        if unique_cards:
            cards_text = ", ".join(unique_cards)
            card_count_text = f"Detected Cards ({len(unique_cards)}): {cards_text}"
            
            cvzone.putTextRect(
                frame, 
                card_count_text, 
                (10, frame.shape[0] - 20), 
                scale=0.7, 
                thickness=1
            )
        
        return frame, unique_cards
    
    def run(self):
        """Run the card detection loop"""
        print("Starting real-time card detection. Press 'q' to quit.")
        
        try:
            while True:
                # Read frame from webcam
                success, frame = self.cap.read()
                if not success:
                    print("Failed to grab frame from webcam")
                    break
                
                # Process the frame
                processed_frame, cards = self.process_frame(frame)
                
                # Display the processed frame
                cv2.imshow(config.WINDOW_NAME, processed_frame)
                
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            # Release resources
            self.cap.release()
            cv2.destroyAllWindows()
            print("Card detection stopped")

def main():
    try:
        detector = CardDetector()
        detector.run()
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()