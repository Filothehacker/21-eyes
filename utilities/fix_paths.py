#!/usr/bin/env python3
"""
Script to fix Ultralytics path issues by creating a proper data.yaml file
with absolute paths and checking directory structure
"""

import os
import yaml
import shutil
import sys

def create_absolute_yaml():
    """Create a data.yaml file with absolute paths"""
    # Get current working directory
    cwd = os.getcwd()
    dataset_dir = os.path.join(cwd, "dataset")
    
    # Define class names
    class_names = [
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
    
    # Create dataset config with absolute paths
    data_dict = {
        # Use absolute paths to avoid path resolution issues
        'train': os.path.join(dataset_dir, "train", "images"),
        'val': os.path.join(dataset_dir, "valid", "images"),
        'test': os.path.join(dataset_dir, "test", "images"),
        'nc': len(class_names),
        'names': class_names
    }
    
    # Write to YAML file
    yaml_path = 'absolute_data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_dict, f, default_flow_style=False)
    
    print(f"Created YAML file with absolute paths: {yaml_path}")
    print(f"Train images path: {data_dict['train']}")
    print(f"Validation images path: {data_dict['val']}")
    print(f"Test images path: {data_dict['test']}")
    
    return yaml_path

def check_and_create_directories():
    """Check if directories exist, create them if needed"""
    cwd = os.getcwd()
    dataset_dir = os.path.join(cwd, "dataset")
    
    if not os.path.exists(dataset_dir):
        print(f"Creating main dataset directory: {dataset_dir}")
        os.makedirs(dataset_dir, exist_ok=True)
    
    # Define required subdirectories
    subdirs = [
        os.path.join(dataset_dir, "train", "images"),
        os.path.join(dataset_dir, "train", "labels"),
        os.path.join(dataset_dir, "valid", "images"),
        os.path.join(dataset_dir, "valid", "labels"),
        os.path.join(dataset_dir, "test", "images"),
        os.path.join(dataset_dir, "test", "labels")
    ]
    
    # Create subdirectories if they don't exist
    for subdir in subdirs:
        if not os.path.exists(subdir):
            print(f"Creating directory: {subdir}")
            os.makedirs(subdir, exist_ok=True)
    
    return True

def check_existing_data():
    """Check if there's existing data in wrong locations and move it if needed"""
    cwd = os.getcwd()
    
    # Check if there are alternative locations
    potential_dirs = [
        os.path.join(cwd, "datasets", "dataset"),
        os.path.join(cwd, "train"),
        os.path.join(cwd, "valid"),
        os.path.join(cwd, "test")
    ]
    
    for dir_path in potential_dirs:
        if os.path.exists(dir_path):
            print(f"Found potential data directory: {dir_path}")
            
            # Check if this is a datasets/dataset structure
            if dir_path.endswith("datasets/dataset"):
                for subdir in ["train", "valid", "test"]:
                    src_img_dir = os.path.join(dir_path, subdir, "images")
                    src_lbl_dir = os.path.join(dir_path, subdir, "labels")
                    
                    dst_img_dir = os.path.join(cwd, "dataset", subdir, "images")
                    dst_lbl_dir = os.path.join(cwd, "dataset", subdir, "labels")
                    
                    if os.path.exists(src_img_dir) and os.listdir(src_img_dir):
                        print(f"Moving images from {src_img_dir} to {dst_img_dir}")
                        for file in os.listdir(src_img_dir):
                            shutil.copy2(
                                os.path.join(src_img_dir, file),
                                os.path.join(dst_img_dir, file)
                            )
                    
                    if os.path.exists(src_lbl_dir) and os.listdir(src_lbl_dir):
                        print(f"Moving labels from {src_lbl_dir} to {dst_lbl_dir}")
                        for file in os.listdir(src_lbl_dir):
                            shutil.copy2(
                                os.path.join(src_lbl_dir, file),
                                os.path.join(dst_lbl_dir, file)
                            )
    
    return True

def create_fixed_train_script():
    """Create a fixed training script"""
    script_content = """
import os
import argparse
from ultralytics import YOLO

def train_model(data_path, model_size='n', epochs=50, imgsz=640, batch=16, project='models'):
    # Initialize model
    model = YOLO(f'yolov8{model_size}.pt')
    
    # Train the model
    try:
        print(f"Training with data config: {data_path}")
        results = model.train(
            data=data_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=project,
            name='card_detection'
        )
        
        print(f"Training completed. Results saved to {results.save_dir}")
        print(f"Best model saved as {os.path.join(results.save_dir, 'weights', 'best.pt')}")
        
        return results
    except Exception as e:
        print(f"Error during training: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 for card detection')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to data.yaml file')
    parser.add_argument('--model_size', type=str, default='n',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--project', type=str, default='models',
                        help='Project directory for saving results')
    
    args = parser.parse_args()
    
    # Train the model
    train_model(
        data_path=args.data,
        model_size=args.model_size,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project
    )

if __name__ == "__main__":
    main()
"""
    
    with open('fixed_train.py', 'w') as f:
        f.write(script_content)
    
    print("Created fixed_train.py script")
    return True

def main():
    print("Starting Ultralytics path fix utility")
    print("=====================================")
    
    # Create directories
    print("\nStep 1: Checking and creating directory structure")
    check_and_create_directories()
    
    # Check for existing data in wrong locations
    print("\nStep 2: Checking for data in incorrect locations")
    check_existing_data()
    
    # Create YAML with absolute paths
    print("\nStep 3: Creating data.yaml with absolute paths")
    yaml_path = create_absolute_yaml()
    
    # Create fixed training script
    print("\nStep 4: Creating fixed training script")
    create_fixed_train_script()
    
    print("\nAll done! You can now run:")
    print(f"python fixed_train.py --data {yaml_path} --epochs 50 --model_size n")

if __name__ == "__main__":
    main()