from imgaug import augmenters as iaa
import cv2
import numpy as np
import random
import os
import gc  # Garbage collector
from tqdm import tqdm

def load_cards(cards_dir):
    print("Loading cards...")
    cards = []
    labels = []
    for filename in tqdm(os.listdir(cards_dir)):
        img_path = os.path.join(cards_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Read the card image
        cards.append(img)
        label = os.path.splitext(filename)[0]
        if label in ("As1", "As2"):
            label = "As"
        labels.append(label)
    print(f"Loaded {len(cards)} cards.")
    return cards, labels


def load_backgrounds(backgrounds_dir, n_backgrounds=1000):
    print(f"Loading up to {n_backgrounds} backgrounds...")
    background_files = os.listdir(backgrounds_dir)
    random.shuffle(background_files)
    # Just return the file paths instead of loading all images into memory
    bg_paths = [os.path.join(backgrounds_dir, f) for f in background_files[:n_backgrounds]]
    print(f"Found {len(bg_paths)} background paths.")
    return bg_paths


def place_card_on_background(card, background):
    # Calculate scaling to fit card on background while maintaining aspect ratio
    card_height, card_width = card.shape[:2]
    bg_height, bg_width = background.shape[:2]
    
    # Ensure card fits within background with some margin
    max_card_height = bg_height - 20  # 10px margin top and bottom
    max_card_width = bg_width - 20    # 10px margin left and right
    
    scale_height = max_card_height / card_height
    scale_width = max_card_width / card_width
    scale = min(scale_height, scale_width, 1.0)  # Don't scale up if card is already smaller
    
    # Resize card maintaining aspect ratio
    if scale < 1.0:  # Only resize if card is too large
        new_height = int(card_height * scale)
        new_width = int(card_width * scale)
        card = cv2.resize(card, (new_width, new_height))
        card_height, card_width = card.shape[:2]
    
    # Randomly position card within background
    max_y_offset = bg_height - card_height
    max_x_offset = bg_width - card_width
    
    if max_y_offset < 0 or max_x_offset < 0:
        print(f"Warning: Card ({card_width}x{card_height}) is larger than background ({bg_width}x{bg_height})")
        return background
    
    y_offset = random.randint(0, max_y_offset)
    x_offset = random.randint(0, max_x_offset)
    
    # Create a copy of the background to avoid modifying the original
    result = background.copy()
    
    # Get alpha channel (normalized to 0-1)
    alpha = card[:, :, 3] > 1
    # Create alpha masks for card and background
    alpha_3channel = np.stack([alpha, alpha, alpha], axis=2)
    inv_alpha_3channel = 1.0 - alpha_3channel
        
    # Extract region from background where card will be placed
    bg_region = result[y_offset:y_offset+card_height, x_offset:x_offset+card_width]
        
    # Blend card RGB with background RGB using alpha mask
    blended = card[:, :, :3] * alpha_3channel + bg_region * inv_alpha_3channel
        
    # Place blended image back into background
    result[y_offset:y_offset+card_height, x_offset:x_offset+card_width] = blended
    return result


def augment_image(image):
    # Pad the image to avoid cropping during rotation
    original_height, original_width = image.shape[:2]
    max_dim = int(np.sqrt(original_height**2 + original_width**2))
    padding_height = (max_dim - original_height) // 2
    padding_width = (max_dim - original_width) // 2
    padded_image = cv2.copyMakeBorder(image, padding_height, padding_height, padding_width, padding_width,
                                       cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))

    rgb_image = padded_image[:, :, :3]
    alpha_channel = padded_image[:, :, 3]
    
    # Make sure the image is in uint8 format
    if rgb_image.dtype != np.uint8:
        if np.max(rgb_image) <= 1.0:
            # If image is normalized between 0-1, rescale to 0-255
            rgb_image = (rgb_image * 255).astype(np.uint8)
        else:
            # Otherwise convert directly
            rgb_image = rgb_image.astype(np.uint8)
    
    degree = random.randint(0, 90)

    # Create an augmentation pipeline for the cards
    card_augmentation = iaa.Sequential([
        iaa.Multiply((0.8, 1.2)),  # Adjust brightness
        iaa.AddToHueAndSaturation((-13, 15)),  # Adjust hue and saturation
        iaa.Add((-10, 10)),  # Adjust contrast
        iaa.GaussianBlur(sigma=(0, 1.0)),  # Apply Gaussian blur
    ])
    
    card_rotation = iaa.Sequential([
        iaa.Affine(rotate=(-degree, degree), scale=(0.5, 1.0)), 
        iaa.Resize((0.45, 0.6))
    ])  # Rotate cards and resize
    
    augmented_rgb = card_augmentation(image=rgb_image)
    card = np.dstack((augmented_rgb, alpha_channel[:, :, np.newaxis]))
    augmented = card_rotation(image=card)

    return augmented


def process_and_save_in_batches(cards, labels, bg_paths, batch_size=200, train=0.6, development=0.2):
    """Process and save images in batches to prevent memory issues"""
    print("Setting up directory structure...")
    data_path = os.path.join(os.getcwd(), "data_classification")
    for split in ["train", "development", "test"]:
        os.makedirs(os.path.join(data_path, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(data_path, split, "labels"), exist_ok=True)
        for file in os.listdir(os.path.join(data_path, split, "images")):
            os.remove(os.path.join(data_path, split, "images", file))
        for file in os.listdir(os.path.join(data_path, split, "labels")):
            os.remove(os.path.join(data_path, split, "labels", file))
    
    # Calculate counts for each split
    total_images = len(cards) * len(bg_paths)
    train_count = int(total_images * train)
    dev_count = int(total_images * development)
    test_count = total_images - train_count - dev_count
    
    print(f"Planned distribution: {train_count} training, {dev_count} development, {test_count} test images")
    
    # Create a shuffled list of all card-background combinations
    all_combinations = []
    for card_idx, label in enumerate(labels):
        for bg_idx in range(len(bg_paths)):
            all_combinations.append((card_idx, bg_idx, label))
    
    random.shuffle(all_combinations)
    
    # Process in batches
    counter = {'train': 0, 'development': 0, 'test': 0}
    limits = {'train': train_count, 'development': dev_count, 'test': test_count}
    current_split = 'train'
    
    print(f"Processing {total_images} images in batches of {batch_size}...")
    
    # Process in batches
    for batch_start in tqdm(range(0, len(all_combinations), batch_size), desc="Batch processing"):
        batch_end = min(batch_start + batch_size, len(all_combinations))
        batch = all_combinations[batch_start:batch_end]
        
        for card_idx, bg_idx, label in tqdm(batch, desc=f"Processing batch {batch_start//batch_size + 1}", leave=False):
            # Determine which split to use
            if counter['train'] < limits['train']:
                split = 'train'
            elif counter['development'] < limits['development']:
                split = 'development'
            else:
                split = 'test'
            
            # Load the background (only when needed)
            bg_path = bg_paths[bg_idx]
            background = cv2.imread(bg_path)
            background = cv2.resize(background, (720, 720))
            
            # Get the card and augment it
            card = cards[card_idx].copy()
            card_aug = augment_image(card)
            
            # Superimpose
            result = place_card_on_background(card_aug, background)
            
            # Save the image
            image_path = os.path.join(data_path, split, "images", f"{counter[split]}.jpg")
            label_path = os.path.join(data_path, split, "labels", f"{counter[split]}.txt")
            
            cv2.imwrite(image_path, result)
            with open(label_path, "w") as f:
                f.write(label)
            
            counter[split] += 1
            
            # Clean up to free memory
            del background, card_aug, result
        
        # Force garbage collection after each batch
        gc.collect()
        
        # Print progress update
        total_processed = sum(counter.values())
        print(f"Progress: {total_processed}/{total_images} images processed. "
              f"Train: {counter['train']}, Dev: {counter['development']}, Test: {counter['test']}")
    
    print("Dataset creation complete!")
    print(f"Final counts - Train: {counter['train']}, Dev: {counter['development']}, Test: {counter['test']}")


if __name__ == "__main__":
    print("Starting playing card dataset generation with memory-efficient processing...")
    cards_dir = os.path.join(os.getcwd(), "data_classification", "52_cards")
    backgrounds_dir = os.path.join(os.getcwd(), "data_classification", "backgrounds")
    
    # Adjust these parameters to fit your memory constraints
    n_backgrounds_per_card = 200  # Reduced from 500
    batch_size = 100  # Process images in smaller batches
    
    cards, labels = load_cards(cards_dir)
    bg_paths = load_backgrounds(backgrounds_dir, n_backgrounds=1000)
    
    # Use a subset of backgrounds per card to reduce total image count
    if len(bg_paths) > n_backgrounds_per_card:
        print(f"Using {n_backgrounds_per_card} backgrounds per card (from {len(bg_paths)} available)")
        bg_paths = random.sample(bg_paths, n_backgrounds_per_card)
    
    # Process and save in batches
    process_and_save_in_batches(cards, labels, bg_paths, batch_size=batch_size)
    
    print("Process completed!")