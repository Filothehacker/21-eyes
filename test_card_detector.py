#file to export to the hpc cluster for the testing
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Card Detection - Testing Script

This script evaluates a trained YOLOv8 model on the test dataset
and generates performance metrics and visualizations.
"""

import os
import argparse
import yaml
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

def plot_confusion_matrix(cm, class_names, output_path, normalize=True):
    """
    Generate a confusion matrix plot.
    """
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
        
    plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=15)
    plt.colorbar()
    
    # Add class labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90, fontsize=8)
    plt.yticks(tick_marks, class_names, fontsize=8)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                plt.text(j, i, format(cm[i, j], '.2f' if normalize else 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black",
                         fontsize=7)
    
    plt.tight_layout()
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test a YOLO model for card detection')
    parser.add_argument('--yaml', type=str, default='hpc_data.yaml', help='Path to the YAML configuration file')
    parser.add_argument('--model', type=str, default='models/best_card_detector.pt', help='Path to the trained model')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--device', type=str, default='', help='Device to use (empty for auto)')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    args = parser.parse_args()

    # Print test information
    print(f"\n{'='*50}")
    print(f"Starting Card Detection Testing")
    print(f"{'='*50}")
    print(f"YAML config:   {args.yaml}")
    print(f"Model:         {args.model}")
    print(f"Image size:    {args.imgsz}x{args.imgsz}")
    print(f"Confidence:    {args.conf}")
    print(f"Batch size:    {args.batch}")
    print(f"Output dir:    {args.output}")
    
    # Check if YAML file exists
    if not os.path.exists(args.yaml):
        raise FileNotFoundError(f"YAML file not found: {args.yaml}")
    
    # Check if model file exists
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    
    # Load and validate YAML file
    with open(args.yaml, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / f"test_results_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print GPU information
    print("\nGPU Information:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device:    {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory:    {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load model
    print(f"\nLoading model: {args.model}")
    try:
        model = YOLO(args.model)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Test parameters
    test_params = {
        'data': args.yaml,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'conf': args.conf,
        'device': args.device if args.device else None,
        'project': str(output_dir),
        'name': 'val_results',
        'exist_ok': True,
        'verbose': True,
        'save_json': True,
        'save_hybrid': True,
        'save_conf': True
    }
    
    # Start testing
    print(f"\n{'='*50}")
    print(f"Starting testing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")
    
    start_time = time.time()
    try:
        # Run validation on test dataset
        results = model.val(**test_params)
        
        # Print summary
        print(f"\n{'='*50}")
        print(f"Testing completed in {(time.time() - start_time):.2f} seconds")
        print(f"mAP@50:     {results.box.map50:.4f}")
        print(f"mAP@50-95:  {results.box.map:.4f}")
        print(f"Precision:  {results.box.mp:.4f}")
        print(f"Recall:     {results.box.mr:.4f}")
        print(f"{'='*50}")
        
        # Generate detailed class-wise metrics
        class_names = yaml_data['names']
        
        # Create per-class performance table
        performance_data = []
        for i, class_name in enumerate(class_names):
            if i < len(results.box.cls_metrics):
                metrics = results.box.cls_metrics[i]
                performance_data.append({
                    'Class': class_name,
                    'Precision': metrics[0],
                    'Recall': metrics[1],
                    'mAP@50': metrics[2],
                    'mAP@50-95': metrics[3]
                })
        
        # Create DataFrame and save to CSV
        if performance_data:
            df = pd.DataFrame(performance_data)
            csv_path = output_dir / 'class_performance.csv'
            df.to_csv(csv_path, index=False)
            print(f"Class performance saved to: {csv_path}")
            
            # Plot class performance metrics
            plt.figure(figsize=(14, 8))
            df.plot(x='Class', y=['Precision', 'Recall', 'mAP@50'], kind='bar', figsize=(14, 8))
            plt.title('Performance Metrics by Card Class')
            plt.ylabel('Score')
            plt.xlabel('Card Class')
            plt.tight_layout()
            plt.savefig(output_dir / 'class_performance.png', dpi=300)
            plt.close()
        
        # Run inference on a few test samples for visualization
        if os.path.exists(yaml_data['test']):
            # Get a few test images
            import glob
            test_images = glob.glob(f"{yaml_data['test']}/*.jpg")[:10]
            
            if test_images:
                print(f"\nGenerating visualizations for {len(test_images)} test images...")
                
                # Create directory for visualizations
                vis_dir = output_dir / 'visualizations'
                vis_dir.mkdir(exist_ok=True)
                
                # Run inference and save visualizations
                for img_path in test_images:
                    img_name = os.path.basename(img_path)
                    results = model.predict(img_path, save=True, imgsz=args.imgsz, conf=args.conf, 
                                           project=str(vis_dir), name='predictions', exist_ok=True)
                
                print(f"Visualizations saved to: {vis_dir}")
        
        # Generate confusion matrix if possible
        try:
            # This is a simplified approach - would need adjustments for your specific dataset
            true_labels = []
            pred_labels = []
            
            # Run predictions on test data to collect true and predicted labels
            test_dir = yaml_data['test']
            if os.path.exists(test_dir):
                # Get ground truth labels from test data
                # You may need to adapt this to your specific dataset structure
                # This is just a placeholder for the concept
                
                cm = confusion_matrix(true_labels, pred_labels)
                cm_plot_path = output_dir / 'confusion_matrix.png'
                plot_confusion_matrix(cm, class_names, cm_plot_path)
                print(f"Confusion matrix saved to: {cm_plot_path}")
        except Exception as e:
            print(f"Could not generate confusion matrix: {e}")
            
        print(f"\nTest results saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()