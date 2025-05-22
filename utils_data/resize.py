import os
from PIL import Image


def resize_images_in_folder(folder_path, new_size=(416, 416)):
    
    # Supported image extensions
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    # Count for summary
    processed_count = 0
    skipped_count = 0
    
    # Process each file in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path):
            continue
            
        # Check if the file is an image
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in supported_formats:
            skipped_count += 1
            continue
            
        try:
            # Open the image and resize it
            with Image.open(file_path) as img:
                resized_img = img.resize(new_size, Image.LANCZOS)
                resized_img.save(file_path, quality=95)
                processed_count += 1
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            skipped_count += 1
    
    print(f"\nResize complete!")
    print(f"Images resized: {processed_count}")
    print(f"Images skipped: {skipped_count}")


if __name__ == "__main__":

    folder_path = input("Enter the folder relative path containing your images: ").strip()
    if not os.path.isdir(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
    else:
        resize_images_in_folder(folder_path, (416, 416))