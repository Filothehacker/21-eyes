from ultralytics import YOLO
model=YOLO("models/card_detection2/weights/best.pt")


import os
import cv2

# Define the correct card names
correct_names = [
    '2C', '2D', '2H', '2S',
    '3C', '3D', '3H', '3S',
    '4C', '4D', '4H', '4S',
    '5C', '5D', '5H', '5S',
    '6C', '6D', '6H', '6S',
    '7C', '7D', '7H', '7S',
    '8C', '8D', '8H', '8S',
    '9C', '9D', '9H', '9S',
    '10C', '10D', '10H', '10S',
    'JC', 'JD', 'JH', 'JS',
    'QC', 'QD', 'QH', 'QS',
    'KC', 'KD', 'KH', 'KS',
    'AC', 'AD', 'AH', 'AS'
]

def predict_with_correct_names(image_path):
    # Read and process the image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get predictions from the model
    results = model(img_rgb)
    
    # Process results with correct names
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            # Get the correct class name using our mapping
            class_name = correct_names[cls_id]
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            detections.append({
                'class': class_name,
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2]
            })
    
    return detections, img

# Test with an image
test_image = "dataset/test/images/IMG_20220316_135316_jpg.rf.73bfd93ba6f1293a44936e5ed9656fa4.jpg"
detections, img = predict_with_correct_names(test_image)

# Print the results
for det in detections:
    print(f"Detected {det['class']} with confidence {det['confidence']:.2f}")

# You can also draw the results on the image
for det in detections:
    x1, y1, x2, y2 = [int(v) for v in det['bbox']]
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f"{det['class']} {det['confidence']:.2f}", 
                (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

# Save or display the image
cv2.imwrite("result.jpg", img)