# Configuration settings for Poker Detector

# Model settings
MODEL_PATH = "models/best_card_detector.pt"    # Path to your trained YOLOv8 model
CONFIDENCE_THRESHOLD = 0.5 # Minimum confidence score for detections

# Camera settings
CAMERA_ID = 0              # Camera ID (0 for default webcam)
CAMERA_WIDTH = 1280        # Camera width resolution
CAMERA_HEIGHT = 720        # Camera height resolution

# UI settings
DRAW_BOUNDING_BOXES = True # Whether to draw boxes around cards
SHOW_CONFIDENCE = True     # Whether to show confidence scores
WINDOW_NAME = "Poker Card Detection"

# Card class names (52 cards)
CLASS_NAMES = [
    '10C', '10D', '10H', '10S',
    '2C', '2D', '2H', '2S',
    '3C', '3D', '3H', '3S',
    '4C', '4D', '4H', '4S',
    '5C', '5D', '5H', '5S',
    '6C', '6D', '6H', '6S',
    '7C', '7D', '7H', '7S',
    '8C', '8D', '8H', '8S',
    '9C', '9D', '9H', '9S',
    'AC', 'AD', 'AH', 'AS',
    'JC', 'JD', 'JH', 'JS',
    'KC', 'KD', 'KH', 'KS',
    'QC', 'QD', 'QH', 'QS'
]

# Colors for visualization (B, G, R format)
COLORS = {
    'bounding_box': (0, 255, 0),     # Green
    'corner_rect': (255, 0, 255),    # Purple
    'text_bg': (0, 200, 255),        # Yellow
    'text': (0, 0, 0)                # Black
}