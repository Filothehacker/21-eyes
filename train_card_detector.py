#file to use in the hpc cluster for the training
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Card Detection - Training Script

This script trains a YOLOv8 model for playing card detection
using the provided YAML configuration file.
"""

import os
import argparse
import yaml
import torch
import time
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a YOLO model for card detection')
    parser.add_argument('--yaml', type=str, default='hpc_data.yaml', help='Path to the YAML configuration file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--model-size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'], help='YOLO model size')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--device', type=str, default='', help='Device to use (empty for auto)')
    args = parser.parse_args()

    # Print training information
    print(f"\n{'='*50}")
    print(f"Starting Card Detection Training")
    print(f"{'='*50}")
    print(f"YAML config:   {args.yaml}")
    print(f"Model size:    YOLOv8{args.model_size}")
    print(f"Epochs:        {args.epochs}")
    print(f"Batch size:    {args.batch}")
    print(f"Image size:    {args.imgsz}x{args.imgsz}")
    print(f"Resume:        {args.resume}")
    
    # Check if YAML file exists
    if not os.path.exists(args.yaml):
        raise FileNotFoundError(f"YAML file not found: {args.yaml}")
    
    # Load and validate YAML file
    with open(args.yaml, 'r') as f:
        yaml_data = yaml.safe_load(f)
        
    # Print dataset information
    print(f"\nDataset information:")
    print(f"- Number of classes: {yaml_data['nc']}")
    print(f"- Training data:     {yaml_data['train']}")
    print(f"- Validation data:   {yaml_data['val']}")
    print(f"- Test data:         {yaml_data['test']}")
    
    # Check if directories exist
    for dir_key in ['train', 'val', 'test']:
        if not os.path.exists(yaml_data[dir_key]):
            print(f"Warning: {dir_key} directory not found: {yaml_data[dir_key]}")

    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"./models/card_detector_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print GPU information
    print("\nGPU Information:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device:    {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory:    {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Initialize model
    pretrained_model = f'yolov8{args.model_size}.pt'
    print(f"\nInitializing model: {pretrained_model}")
    
    try:
        model = YOLO(pretrained_model)
        print("Model initialized successfully")
    except Exception as e:
        print(f"Error initializing model: {e}")
        return
    
    # Copy YAML file to output directory for reference
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(yaml_data, f)
    
    # Training parameters
    train_params = {
        'data': args.yaml,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'patience': 15,
        'project': str(output_dir.parent),
        'name': output_dir.name,
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'auto',
        'verbose': True,
        'save': True,
        'save_period': 10,  # Save a checkpoint every 10 epochs
        'device': args.device if args.device else None
    }
    
    if args.resume:
        # Find the latest checkpoint
        checkpoints = list(output_dir.glob('weights/epoch*.pt'))
        if checkpoints:
            latest_checkpoint = str(max(checkpoints, key=os.path.getctime))
            print(f"Resuming training from checkpoint: {latest_checkpoint}")
            train_params['resume'] = latest_checkpoint
        else:
            print("No checkpoints found, starting from scratch")
    
    # Start training
    print(f"\n{'='*50}")
    print(f"Starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")
    
    start_time = time.time()
    try:
        # Execute training
        results = model.train(**train_params)
        
        # Save best model to models folder
        best_model_path = output_dir / 'weights' / 'best.pt'
        final_model_path = Path('models') / 'best_card_detector.pt'
        Path('models').mkdir(exist_ok=True)
        
        if os.path.exists(best_model_path):
            import shutil
            shutil.copy(best_model_path, final_model_path)
            print(f"\nBest model copied to: {final_model_path}")
        
        # Print training summary
        print(f"\n{'='*50}")
        print(f"Training completed in {(time.time() - start_time) / 60:.2f} minutes")
        print(f"Best mAP@50: {results.maps[50]:.4f}")
        print(f"Best mAP@50-95: {results.box.map:.4f}")
        print(f"Results saved to {output_dir}")
        print(f"{'='*50}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    
if __name__ == "__main__":
    main()