import cv2
import os
import torch
import numpy as np
import yaml
from ultralytics import YOLO
import matplotlib.pyplot as plt
import random
import glob
import matplotlib.patches as patches


def visualize_pred(image, boxes, class_ids, classes, confidences=None):
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    if len(boxes) > 0:
        for i, (box, class_id) in enumerate(zip(boxes, class_ids)):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            
            label = classes[class_id] if class_id < len(classes) else f"Class {class_id}"
            if confidences is not None:
                label += f" {confidences[i]:.2f}"
            
            ax.text(x1, y1-5, label, fontsize=10, color='red', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    ax.set_title("YOLOv8 Detections")
    ax.axis('off')
    plt.show()


if __name__ == "__main__":
    cwd = os.getcwd()
    
    # Load classes
    classes_path = os.path.join(cwd, "configurations", "classes.yaml")
    with open(classes_path, "r") as f:
        classes = yaml.safe_load(f)
    CLASSES = classes["classes"]
    
    # Load model
    model_path = '/Users/filippofocaccia/Desktop/new-21-eyes/v8/yolov8_mio.pt'
    model = YOLO(model_path)
    
    # Find and select random image
    test_images_dir = os.path.join(cwd, "data_yolo", "test", "images")
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    all_images = []
    for ext in image_extensions:
        all_images.extend(glob.glob(os.path.join(test_images_dir, ext)))
        all_images.extend(glob.glob(os.path.join(test_images_dir, ext.upper())))
    
    image_path = random.choice(all_images)
    image_filename = os.path.basename(image_path)
    
    print(f"Testing image: {image_filename}")
    
    # Load image
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Run inference
    results = model(image_path, verbose=False)
    
    # Extract results
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        
        print(f"Found {len(boxes)} detections")
        
        # Visualize
        visualize_pred(
            image=image_rgb,
            boxes=boxes,
            class_ids=class_ids,
            classes=CLASSES,
            confidences=confidences
        )
    else:
        print("No detections found")
        # Show image anyway
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image_rgb)
        ax.set_title("No Detections")
        ax.axis('off')
        plt.show()