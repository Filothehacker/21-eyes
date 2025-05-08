import os
from PIL import Image

def resize_images_in_folder(folder_path, new_size=(416, 416)):
    """
    Resize all images in a folder to the specified size and replace the originals.
    
    Args:
        folder_path: Path to the folder containing images
        new_size: Tuple of (width, height) for the new image size
    """
    # Supported image extensions
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    # Count for summary
    processed_count = 0
    skipped_count = 0
    
    # Process each file in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Skip if it's not a file
        if not os.path.isfile(file_path):
            continue
            
        # Check if the file is an image
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in supported_formats:
            skipped_count += 1
            continue
            
        try:
            # Open the image
            with Image.open(file_path) as img:
                # Resize with high quality
                resized_img = img.resize(new_size, Image.LANCZOS)
                
                # Save and replace the original
                resized_img.save(file_path, quality=95)
                
                processed_count += 1
                # print(f"Resized: {filename}")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            skipped_count += 1
    
    # Print summary
    print(f"\nResize complete!")
    print(f"Images resized: {processed_count}")
    print(f"Files skipped: {skipped_count}")

# Example usage
if __name__ == "__main__":
    # Get folder path from user
    folder_path = input("Enter the folder path containing your images: ").strip()
    
    # Check if the folder exists
    if not os.path.isdir(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist or is not a directory.")
    else:
        # Ask for confirmation
        print(f"This will resize all images in '{folder_path}' to 416x416 and REPLACE the originals.")
        confirmation = input("Do you want to continue? (yes/no): ").strip().lower()
        
        if confirmation in ['yes', 'y']:
            resize_images_in_folder(folder_path, (416, 416))
        else:
            print("Operation canceled.")