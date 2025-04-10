
#!/usr/bin/env python3
"""
Script to verify the dataset directory structure and show paths for troubleshooting
"""

import os
import sys
import yaml

def check_dir(path, description):
    """Check if directory exists and print status"""
    exists = os.path.isdir(path)
    status = "✓" if exists else "✗"
    print(f"  {status} {description}: {path}")
    if not exists:
        print(f"    Directory missing: {path}")
    else:
        # Count files
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        print(f"    Found {len(files)} files")
    return exists

def main():
    # Check current directory
    print(f"Current working directory: {os.getcwd()}")
    
    # Check if dataset directory exists
    dataset_dir = os.path.join(os.getcwd(), "dataset")
    print(f"\nChecking main dataset directory:")
    if not check_dir(dataset_dir, "Main dataset directory"):
        print("Main dataset directory not found. Please create it first.")
        sys.exit(1)
    
    # Check subdirectories
    print("\nChecking required subdirectories:")
    required_dirs = [
        ("train/images", "Training images"),
        ("valid/images", "Validation images"),
        ("test/images", "Test images"),
        ("train/labels", "Training labels"),
        ("valid/labels", "Validation labels"),
        ("test/labels", "Test labels")
    ]
    
    all_exist = True
    for subdir, desc in required_dirs:
        path = os.path.join(dataset_dir, subdir)
        if not check_dir(path, desc):
            all_exist = False
    
    # Check data.yaml file
    print("\nChecking data.yaml file:")
    yaml_path = os.path.join(os.getcwd(), "data.yaml")
    if os.path.isfile(yaml_path):
        print(f"  ✓ data.yaml file found: {yaml_path}")
        
        # Validate YAML content
        try:
            with open(yaml_path, 'r') as f:
                yaml_data = yaml.safe_load(f)
            
            # Check required keys
            required_keys = ['path', 'train', 'val', 'test', 'nc', 'names']
            missing_keys = [key for key in required_keys if key not in yaml_data]
            
            if missing_keys:
                print(f"  ✗ Missing required keys in data.yaml: {missing_keys}")
            else:
                print("  ✓ All required keys present in data.yaml")
                
                # Check if paths are valid
                base_path = yaml_data['path']
                train_path = os.path.join(base_path, yaml_data['train']) 
                val_path = os.path.join(base_path, yaml_data['val'])
                test_path = os.path.join(base_path, yaml_data['test'])
                
                # Replace ./ with current directory if present
                if base_path.startswith('./'):
                    base_path = os.path.join(os.getcwd(), base_path[2:])
                
                print("\nPath resolution from data.yaml:")
                print(f"  Base path: {base_path}")
                print(f"  Train images: {os.path.join(base_path, yaml_data['train'])}")
                print(f"  Validation images: {os.path.join(base_path, yaml_data['val'])}")
                print(f"  Test images: {os.path.join(base_path, yaml_data['test'])}")
                
                # Check if number of classes matches
                num_names = len(yaml_data['names'])
                if num_names != yaml_data['nc']:
                    print(f"  ✗ Mismatch between nc ({yaml_data['nc']}) and names list length ({num_names})")
                else:
                    print(f"  ✓ Number of classes matches names list ({num_names})")
            
        except Exception as e:
            print(f"  ✗ Error parsing data.yaml: {e}")
    else:
        print(f"  ✗ data.yaml file not found: {yaml_path}")
    
    # Summary
    print("\nSummary:")
    if all_exist:
        print("✓ Directory structure looks good!")
    else:
        print("✗ Some directories are missing, please create them.")
    
    # Provide commands to run
    print("\nTo train your model, run:")
    print("python fixed_train.py --data data.yaml --epochs 50 --model_size n")

if __name__ == "__main__":
    main()