
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Get command line arguments
img_path = sys.argv[1]
results_path = sys.argv[2]
class_names = sys.argv[3].split(',')
conf_thres = float(sys.argv[4])

# Load image
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Load results if they exist
if os.path.exists(results_path):
    with open(results_path, 'r') as f:
        lines = f.readlines()
    
    # Draw each detection
    for line in lines:
        values = line.strip().split()
        if len(values) >= 6:  # class x y w h conf
            cls_id = int(values[0])
            cls_name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
            
            # Convert from YOLO format to pixel coordinates
            x, y, w, h = map(float, values[1:5])
            conf = float(values[5]) if len(values) > 5 else 1.0
            
            if conf < conf_thres:
                continue
                
            H, W = img.shape[:2]
            x1, y1 = int((x - w/2) * W), int((y - h/2) * H)
            x2, y2 = int((x + w/2) * W), int((y + h/2) * H)
            
            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Add text with class name
            text = f"{cls_name}: {conf:.2f}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(img, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), (0, 0, 255), -1)
            cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Save the image
output_path = os.path.join(os.path.dirname(results_path), os.path.basename(img_path).rsplit('.', 1)[0] + '_labeled.jpg')
plt.figure(figsize=(12, 12))
plt.imshow(img)
plt.axis('off')
plt.title('Detections with Class Names')
plt.tight_layout()
plt.savefig(output_path)
print(f"Saved labeled image to {output_path}")
