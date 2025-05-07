import os
import argparse
import yaml
import torch
import time
from datetime import datetime
from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO

def plot_images_grid(images, labels, preds, class_names, conf_thres=0.25, max_imgs=16, figsize=(15, 15)):
    """Plot a grid of images with ground truth and predicted bounding boxes"""
    n = min(len(images), max_imgs)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    axs = axs.flatten() if n > 1 else [axs]
    
    for i, (img, lbl, pred) in enumerate(zip(images[:n], labels[:n], preds[:n])):
        if img is None:
            continue
            
        ax = axs[i]
        ax.imshow(img)
        
        # Plot ground truth boxes (green)
        if lbl is not None:
            for box in lbl:
                # YOLO format: [cls, x, y, w, h] where x,y,w,h are normalized
                cls_id, x, y, w, h = box
                cls_name = class_names[int(cls_id)] if int(cls_id) < len(class_names) else f"Class {int(cls_id)}"
                x1, y1 = (x - w/2) * img.shape[1], (y - h/2) * img.shape[0]
                x2, y2 = (x + w/2) * img.shape[1], (y + h/2) * img.shape[0]
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='lime', linewidth=2)
                ax.add_patch(rect)
                ax.text(x1, y1-10, f"{cls_name}", color='black', 
                      bbox=dict(facecolor='lime', alpha=0.5))
        
        # Plot predicted boxes (blue)
        if pred is not None:
            for det in pred:
                # YOLOv5 format: [x1, y1, x2, y2, conf, cls]
                if det[4] > conf_thres:  # Filter by confidence threshold
                    x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
                    cls_id = int(det[5])
                    cls_name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='blue', linewidth=2)
                    ax.add_patch(rect)
                    ax.text(x1, y1, f"{cls_name}: {det[4]:.2f}", color='white', 
                            bbox=dict(facecolor='blue', alpha=0.5))
        
        ax.set_title(f"Image {i+1}")
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(n, len(axs)):
        axs[i].axis('off')
    
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description='Test a trained YOLOv5 model')
    parser.add_argument('--weights', type=str, required=True, help='Path to the trained weights')
    parser.add_argument('--yaml', type=str, default='classes_v5.yaml', help='Path to the dataset YAML file')
    parser.add_argument('--img-size', type=int, default=448, help='Size of input images')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold for detections')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IoU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=300, help='Maximum number of detections per image')
    parser.add_argument('--device', type=str, default='', help='Device to use (empty for auto)')
    parser.add_argument('--save-txt', action='store_true', help='Save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='Save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='Save cropped prediction boxes')
    parser.add_argument('--save-img', action='store_true', help='Save images with detections')
    parser.add_argument('--visualize', action='store_true', help='Visualize features')
    parser.add_argument('--classes', nargs='+', type=int, help='Filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='Class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='Augmented inference')
    parser.add_argument('--half', action='store_true', help='Use FP16 half-precision inference')
    args = parser.parse_args()

    # Print testing information
    print(f"\n{'='*50}")
    print(f"Starting YOLOv5 Model Evaluation")
    print(f"{'='*50}")
    print(f"Model weights: {args.weights}")
    print(f"YAML config:   {args.yaml}")
    print(f"Image size:    {args.img_size}x{args.img_size}")
    print(f"Confidence:    {args.conf_thres}")
    print(f"IoU threshold: {args.iou_thres}")
    
    # Check if weights exist
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Model weights not found: {args.weights}")
    
    # Check if YAML file exists
    if not os.path.exists(args.yaml):
        raise FileNotFoundError(f"YAML file not found: {args.yaml}")
    
    # Load YAML file
    with open(args.yaml, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    # Get class names from YAML
    if 'names' in yaml_data:
        class_names = yaml_data['names']
    elif 'classes' in yaml_data:
        class_names = yaml_data['classes']
    else:
        # Create generic class names if not found
        class_names = [f"Class {i}" for i in range(yaml_data['nc'])]
        print("Warning: No class names found in YAML file. Using generic class names.")
    
    # Check for test path
    if 'test' not in yaml_data:
        print("Warning: 'test' path not found in YAML file. Using 'val' path instead.")
        test_path = yaml_data.get('val')
    else:
        test_path = yaml_data.get('test')
    
    if not test_path or not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data path not found: {test_path}")
    
    # Print class information
    print(f"\nDataset information:")
    print(f"- Number of classes: {yaml_data['nc']}")
    print(f"- Classes: {class_names}")
    print(f"- Test data: {test_path}")
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"./runs/test/test_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print GPU information
    print("\nGPU Information:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device:    {torch.cuda.get_device_name(0)}")
    
    # Load model
    print(f"\nLoading model from {args.weights}...")
    try:
        model = YOLO(args.weights)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Copy YAML to output directory for reference
    with open(output_dir / 'dataset.yaml', 'w') as f:
        yaml.dump(yaml_data, f)
    
    # Start testing
    print(f"\n{'='*50}")
    print(f"Starting evaluation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")
    
    start_time = time.time()
    try:
        # Run validation using the validate method
        results = model.val(
            data=args.yaml,
            imgsz=args.img_size,
            batch=16,
            device=args.device if args.device else None,
            conf=args.conf_thres,
            iou=args.iou_thres,
            max_det=args.max_det,
            save_json=True,
            save_hybrid=True,
            save_conf=args.save_conf,
            save_txt=args.save_txt,
            save_crop=args.save_crop,
            plots=True,
            half=args.half,
            project=str(output_dir.parent),
            name=output_dir.name,
            exist_ok=True
        )
        
        # Extract metrics
        metrics = results

        # Print results
        print(f"\n{'='*50}")
        print(f"Evaluation completed in {(time.time() - start_time):.2f} seconds")
        print(f"{'='*50}")
        print(f"Results:")
        
        # Try to get detailed metrics - format may vary between versions
        try:
            print(f"mAP@0.5: {metrics.box.map50:.4f}")
            print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
            print(f"Precision: {metrics.box.mp:.4f}")
            print(f"Recall: {metrics.box.mr:.4f}")
        except (AttributeError, TypeError):
            print("Metrics format varies. See complete results in console output above.")
        
        print(f"\nResults saved to {output_dir}")
        
        # Copy visualization plots to output directory
        try:
            results_plots = list(Path('runs/val').glob('**/results.png'))
            if results_plots:
                latest_plot = max(results_plots, key=os.path.getctime)
                shutil.copy(latest_plot, output_dir / 'results.png')
                print(f"Results plot saved to {output_dir / 'results.png'}")
        except Exception as e:
            print(f"Warning: Could not copy results plot: {e}")
        
        # Optional: Run inference on a few test images and visualize results
        print("\nRunning inference on test images for visualization...")
        
        # Get some test images
        images_folder = Path(test_path) / 'images' if (Path(test_path) / 'images').exists() else Path(test_path)
        test_images = list(images_folder.glob('*.jpg')) + list(images_folder.glob('*.png'))
        
        if test_images:
            # Select up to 16 random images
            if len(test_images) > 16:
                import random
                test_images = random.sample(test_images, 16)
            
            # Run inference and visualize
            vis_output_dir = output_dir / 'visualizations'
            vis_output_dir.mkdir(exist_ok=True)
            
            # Create a custom visualization script to display class names
            with open(vis_output_dir / 'custom_vis.py', 'w') as f:
                f.write("""
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
""")
            
            for i, img_path in enumerate(test_images):
                # Run inference
                results = model.predict(
                    source=img_path,
                    imgsz=args.img_size,
                    conf=args.conf_thres,
                    iou=args.iou_thres,
                    max_det=args.max_det,
                    save=True,
                    save_txt=True,
                    save_conf=True,
                    project=str(vis_output_dir),
                    name=f"img_{i}",
                    exist_ok=True
                )
                
                # Save original image with predictions overlaid
                if args.save_img:
                    # Find the labels file
                    labels_dir = vis_output_dir / f"img_{i}" / "labels"
                    if labels_dir.exists():
                        label_files = list(labels_dir.glob(f"{img_path.stem}*.txt"))
                        if label_files:
                            # Run custom visualization
                            cmd = f"python {vis_output_dir / 'custom_vis.py'} {img_path} {label_files[0]} {','.join(class_names)} {args.conf_thres}"
                            os.system(cmd)
                            print(f"Created custom visualization for {img_path.name}")
            
            print(f"Visualizations saved to {vis_output_dir}")
        
        # Create a summary markdown with examples and results
        with open(output_dir / 'summary.md', 'w') as f:
            f.write(f"# YOLOv5 Model Evaluation Summary\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## Model Information\n")
            f.write(f"- **Weights:** {args.weights}\n")
            f.write(f"- **Image Size:** {args.img_size}x{args.img_size}\n")
            f.write(f"- **Confidence Threshold:** {args.conf_thres}\n")
            f.write(f"- **IoU Threshold:** {args.iou_thres}\n\n")
            
            f.write(f"## Dataset Information\n")
            f.write(f"- **Number of Classes:** {yaml_data['nc']}\n")
            f.write(f"- **Classes:** {', '.join(class_names)}\n")
            f.write(f"- **Test Data Path:** {test_path}\n\n")
            
            f.write(f"## Performance Metrics\n")
            try:
                f.write(f"- **mAP@0.5:** {metrics.box.map50:.4f}\n")
                f.write(f"- **mAP@0.5:0.95:** {metrics.box.map:.4f}\n")
                f.write(f"- **Precision:** {metrics.box.mp:.4f}\n")
                f.write(f"- **Recall:** {metrics.box.mr:.4f}\n\n")
            except (AttributeError, TypeError):
                f.write(f"- See console output for detailed metrics\n\n")
            
            f.write(f"## Visualization\n")
            f.write(f"Example detections are available in the `visualizations` directory.\n")
            f.write(f"Results plots are available in the main output directory.\n")
        
        print(f"\n{'='*50}")
        print(f"Testing complete. Results available at {output_dir}")
        print(f"{'='*50}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()