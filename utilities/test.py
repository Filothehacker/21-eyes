import os
import cv2
import numpy as np
from ultralytics import YOLO
from glob import glob
import argparse
import time

def create_grid_image(images, labels, cols=4, cell_size=(320, 320), padding=10):
    """Create a grid of images with their labels"""
    # Calculate rows needed
    n_images = len(images)
    rows = (n_images + cols - 1) // cols
    
    # Create a blank canvas
    canvas_width = cols * (cell_size[0] + padding) + padding
    canvas_height = rows * (cell_size[1] + padding + 30) + padding  # Extra 30px for text
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    
    # Place images in grid
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= n_images:
                break
                
            # Calculate position
            x = col * (cell_size[0] + padding) + padding
            y = row * (cell_size[1] + padding + 30) + padding
            
            # Resize image to fit cell
            img = cv2.resize(images[idx], cell_size)
            
            # Place image
            canvas[y:y+cell_size[1], x:x+cell_size[0]] = img
            
            # Add label
            cv2.putText(
                canvas, 
                labels[idx], 
                (x, y+cell_size[1]+20), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (0, 0, 0), 
                1
            )
            
            idx += 1
    
    return canvas

def test_model_on_folder(model_path, images_folder, output_folder, conf_threshold=0.25):
    """Test YOLOv8 model on all images in a folder"""
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load YOLOv8 model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(images_folder, ext)))
        image_files.extend(glob(os.path.join(images_folder, ext.upper())))
    
    if not image_files:
        print(f"No image files found in {images_folder}")
        return
    
    print(f"Found {len(image_files)} images. Starting detection...")
    
    # Lists to store results for summary
    all_results = []
    processed_images = []
    result_labels = []
    detection_times = []
    
    # Process each image
    for i, img_path in enumerate(image_files):
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue
        
        original_img = img.copy()
        
        # Get file name without extension
        file_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Record detection time
        start_time = time.time()
        
        # Run detection
        results = model(img)
        
        # Calculate detection time
        detection_time = time.time() - start_time
        detection_times.append(detection_time)
        
        # Create a copy of the image to draw on
        result_img = results[0].plot()
        
        # Store the original image and the prediction
        processed_images.append(result_img)
        
        # Get detection results
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = model.names.get(cls_id, f"Class {cls_id}")
                
                if conf >= conf_threshold:
                    detections.append((cls_name, conf))
        
        # Sort detections by confidence (highest first)
        detections.sort(key=lambda x: x[1], reverse=True)
        
        # Create result label
        if detections:
            top_detection = detections[0]
            result_label = f"{file_name}: {top_detection[0]} ({top_detection[1]:.2f})"
        else:
            result_label = f"{file_name}: No detection"
        
        result_labels.append(result_label)
        
        # Save detailed result
        all_results.append({
            "file_name": file_name,
            "detections": detections,
            "detection_time": detection_time
        })
        
        # Save individual result image
        output_path = os.path.join(output_folder, f"{file_name}_detection.jpg")
        cv2.imwrite(output_path, result_img)
        
        print(f"Processed {i+1}/{len(image_files)}: {result_label}")
    
    # Create a summary image with all results in a grid
    print("Creating summary image...")
    summary_img = create_grid_image(processed_images, result_labels)
    summary_path = os.path.join(output_folder, "summary_grid.jpg")
    cv2.imwrite(summary_path, summary_img)
    
    # Create a summary text file
    summary_text_path = os.path.join(output_folder, "detection_summary.txt")
    with open(summary_text_path, 'w') as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"Confidence threshold: {conf_threshold}\n")
        f.write(f"Total images processed: {len(all_results)}\n")
        f.write(f"Average detection time: {sum(detection_times)/len(detection_times):.4f} seconds\n\n")
        
        f.write("DETECTION RESULTS:\n")
        f.write("=" * 50 + "\n")
        
        for result in all_results:
            f.write(f"\nFile: {result['file_name']}\n")
            f.write(f"Detection time: {result['detection_time']:.4f} seconds\n")
            
            if result['detections']:
                f.write("Detections (sorted by confidence):\n")
                for i, (cls_name, conf) in enumerate(result['detections']):
                    f.write(f"  {i+1}. {cls_name}: {conf:.4f}\n")
            else:
                f.write("No detections found.\n")
            
            f.write("-" * 30 + "\n")
    
    print(f"Detection complete! Results saved to {output_folder}")
    print(f"Summary grid image: {summary_path}")
    print(f"Detailed summary: {summary_text_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test YOLOv8 model on a folder of images")
    parser.add_argument("--model", type=str, default="models/best_card_detector.pt", help="Path to YOLOv8 model file")
    parser.add_argument("--input", type=str, default="dataset/test/images", help="Folder containing card images")
    parser.add_argument("--output", type=str, default="detection_results", help="Folder to save detection results")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (0-1)")
    
    args = parser.parse_args()
    
    test_model_on_folder(args.model, args.input, args.output, args.conf)