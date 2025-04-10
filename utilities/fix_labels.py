import os
import glob

# Directory containing label files
label_dirs = [
    "/Users/filippofocaccia/Desktop/card_detector/dataset/train/labels",
    "/Users/filippofocaccia/Desktop/card_detector/dataset/valid/labels",
    "/Users/filippofocaccia/Desktop/card_detector/dataset/test/labels"
]

# Maximum valid class ID (based on your 52 classes)
max_class_id = 51

for label_dir in label_dirs:
    if not os.path.exists(label_dir):
        continue
    
    # Get all txt files in the directory
    label_files = glob.glob(f"{label_dir}/*.txt")
    
    for file_path in label_files:
        fixed_lines = []
        has_invalid_class = False
        
        # Read the file
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Check each line for invalid class IDs
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
                
            class_id = int(parts[0])
            
            # If class ID is invalid, fix it or skip it
            if class_id > max_class_id:
                has_invalid_class = True
                # Either skip this line or map to a valid class
                # Option 1: Skip this object annotation
                continue
                
                # Option 2: Map to a valid class (e.g., to class 0)
                # parts[0] = "0"
                # fixed_lines.append(" ".join(parts) + "\n")
            else:
                fixed_lines.append(line)
        
        # Write back fixed content if needed
        if has_invalid_class:
            with open(file_path, 'w') as f:
                f.writelines(fixed_lines)
            print(f"Fixed invalid class IDs in {file_path}")