import albumentations
import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt


def load_cards(cards_dir):

    cards = []
    labels = []
    for filename in os.listdir(cards_dir):
        img_path = os.path.join(cards_dir, filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        cards.append(img)
        label = os.path.splitext(filename)[0]
        if label in ("As1", "As2"):
            label = "As"
        labels.append(label)
    return cards, labels


def load_backgrounds(backgrounds_dir, n_backgrounds=10):

    backgrounds = []
    background_files = os.listdir(backgrounds_dir)
    random.shuffle(background_files)
    for i, filename in enumerate(background_files):
        if i >= n_backgrounds:
            break
        img_path = os.path.join(backgrounds_dir, filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        backgrounds.append(img)
    return backgrounds


def place_card_on_background(card_img, background_img, card_relative_size=(0.35, 0.5), output_size=(448, 448)):

    # Resize the background
    background_img = cv2.resize(background_img, output_size, interpolation=cv2.INTER_AREA)

    # Scale the card while maintaining aspect ratio
    relative_size = random.uniform(card_relative_size[0], card_relative_size[1])
    card_h, card_w = card_img.shape[:2]
    scale_factor = min(output_size[0] * relative_size / card_w, output_size[1] * relative_size / card_h)
    card_w = int(card_w * scale_factor)
    card_h = int(card_h * scale_factor)
    resized_card = cv2.resize(card_img, (card_w, card_h))

    # Split the card into RGB and alpha channels
    card_rgb = resized_card[:, :, :3]
    card_alpha = resized_card[:, :, 3]

    # Randomize the position of the card on the background and ensure it fits
    x_offset = random.randint(0, output_size[0] - card_w)
    y_offset = random.randint(0, output_size[1] - card_h)

    # Superimpose the card on the background using the alpha channel
    for c in range(3):  # Loop over RGB channels
        background_img[y_offset:y_offset+card_h, x_offset:x_offset+card_w, c] = (
            card_rgb[:, :, c] * (card_alpha) +
            background_img[y_offset:y_offset+card_h, x_offset:x_offset+card_w, c] * (1 - card_alpha)
        )

    return background_img


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
    degree = random.randint(-90, 90)

    # Apply transformations only to the RGB channels
    transform = albumentations.Compose([
        albumentations.RandomBrightnessContrast(brightness_limit=(0.2, 0.4), contrast_limit=(0.2, 0.4), p=0.5),
        albumentations.GaussianBlur(blur_limit=(3, 7), p=0.5),
        albumentations.RandomGamma(gamma_limit=100, p=0.5),
        albumentations.HueSaturationValue(hue_shift_limit=(-50, 50), sat_shift_limit=(-50, 50), val_shift_limit=(-50, 50), p=0.5),
    ])
    augmented_rgb = transform(image=rgb_image)["image"]
    augmented = np.dstack((augmented_rgb, alpha_channel))

    rotate = albumentations.Rotate(limit=(degree, degree), p=1)
    augmented = rotate(image=augmented)["image"]

    plt.imshow(augmented)
    return augmented


def superimpose_images(cards_dir, backgrounds_dir, n_backgrounds=1):

    cards, labels = load_cards(cards_dir)
    backgrounds = load_backgrounds(backgrounds_dir)
    aug_images = []
    aug_labels = []

    for card, label in zip(cards, labels):
        backgrounds_sample = random.sample(backgrounds, n_backgrounds)
        for background in backgrounds_sample:

            # Convert data types
            background = (background/255).astype(np.float32)
            card = (card/255).astype(np.float32)
            card = augment_image(card)

            # Superimpose the card on the background
            image = place_card_on_background(card, background)
            aug_images.append(image)
            aug_labels.append(label)

    return aug_images, aug_labels


def save_dataset(images, labels, train=0.6, development=0.2):

    # Create directories if they do not exist and clean them if they do
    data_path = os.path.join(os.getcwd(), "data_classification")
    for split in ["train", "development", "test"]:
        os.makedirs(os.path.join(data_path, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(data_path, split, "labels"), exist_ok=True)
        for file in os.listdir(os.path.join(data_path, split, "images")):
            os.remove(os.path.join(data_path, split, "images", file))
        for file in os.listdir(os.path.join(data_path, split, "labels")):
            os.remove(os.path.join(data_path, split, "labels", file))

    # Shuffle the data and compute the sizes
    idxs = list(range(len(images)))
    random.shuffle(idxs)
    train_size = int(len(images) * train)
    development_size = int(len(images) * development)

    # Save the train set
    train_idxs = idxs[:train_size]
    train_images = [images[i] for i in train_idxs]
    train_labels = [labels[i] for i in train_idxs]
    for i, (image, label) in enumerate(zip(train_images, train_labels)):
        cv2.imwrite(os.path.join(data_path, "train", "images", f"{i}.jpg"), cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        with open(os.path.join(data_path, "train", "labels", f"{i}.txt"), "w") as f:
            f.write(label)

    # Save the development set
    development_idxs = idxs[train_size:train_size + development_size]
    development_images = [images[i] for i in development_idxs]
    development_labels = [labels[i] for i in development_idxs]
    for i, (image, label) in enumerate(zip(development_images, development_labels)):
        cv2.imwrite(os.path.join(data_path, "development", "images", f"{i}.jpg"), image)
        with open(os.path.join(data_path, "development", "labels", f"{i}.txt"), "w") as f:
            f.write(label)

    # Save the test set
    test_idxs = idxs[train_size + development_size:]
    test_images = [images[i] for i in test_idxs]
    test_labels = [labels[i] for i in test_idxs]
    for i, (image, label) in enumerate(zip(test_images, test_labels)):
        cv2.imwrite(os.path.join(data_path, "test", "images", f"{i}.jpg"), image)
        with open(os.path.join(data_path, "test", "labels", f"{i}.txt"), "w") as f:
            f.write(label)


if __name__ == "__main__":
    
    cards_dir = os.path.join(os.getcwd(), "data_classification", "52_cards")
    backgrounds_dir = os.path.join(os.getcwd(), "data_classification", "backgrounds")

    images, labels = superimpose_images(cards_dir, backgrounds_dir)
    save_dataset(images, labels)