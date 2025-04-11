import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def precompute_img_stats(images_dir):
    """
    Precompute the mean and standard deviation of the images in the given directory.
    This is useful for normalizing the images before training a model.

    :param images_dir: Directory with the images
    :return: Tuple of mean and std
    """

    mean = np.zeros(3)
    std = np.zeros(3)
    num_images = 0

    for image_file in sorted(os.listdir(images_dir)):
        if not image_file.endswith((".jpg", ".jpeg", ".png")):
            continue

        # Load the image and convert it to RGB
        img_path = os.path.join(images_dir, image_file)
        img = Image.open(img_path).convert("RGB")
        # img = img.resize((448, 448))

        # Compute the stats
        img = np.array(img)/255.0
        mean += img.mean(axis=(0, 1))
        std += img.std(axis=(0, 1))
        num_images += 1
    
    mean /= num_images
    std /= num_images
    return mean, std


class CustomDataset(Dataset):
    def __init__(self, images_dir, labels_dir, classes, transform=None, input_size=(448, 448)):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.classes = classes
        self.transform = transform
        self.input_size = input_size
        
        self.image_files = []
        self.label_files = []
        
        for image_file in sorted(os.listdir(images_dir)):
            if not image_file.endswith((".jpg", ".jpeg", ".png")):
                continue
                
            label_file = os.path.splitext(image_file)[0] + ".txt"
            if not os.path.exists(os.path.join(labels_dir, label_file)):
                continue
                
            self.image_files.append(image_file)
            self.label_files.append(label_file)
    

    def __len__(self):
        return len(self.image_files)
    

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        img = Image.open(img_path).convert("RGB")
        
        # Apply image transformations if provided or default to basic transformations
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.Compose([
                # transforms.Resize(self.input_size),
                transforms.ToTensor(),
                # transforms.Normalize(
                #     mean=[0.485, 0.456, 0.406],
                #     std=[0.229, 0.224, 0.225]
                # )
            ])(img)

        label_path = os.path.join(self.labels_dir, self.label_files[idx])
        boxes = []
        classes = []
        
        with open(label_path, "r") as f:
            for line in f:
                values = line.strip().split()
                if len(values) != 5:
                    continue
                    
                class_id, x_center, y_center, width, height = map(float, values)
                class_id = int(class_id)
                
                # Extract rank and suit from class_id or class name
                class_name = self.classes[class_id]
                if class_name.startswith("10"):
                    rank = "10"
                    suit = class_name[2]
                else:
                    rank = class_name[0]
                    suit = class_name[1]
                
                # Map rank to index
                rank_map = {"2": 0, "3": 1, "4": 2, "5": 3, "6": 4, "7": 5, "8": 6, 
                           "9": 7, "10": 8, "J": 9, "Q": 10, "K": 11, "A": 12}
                rank_idx = rank_map.get(rank, 0)
                
                # Map suit to index
                suit_map = {"C": 0, "D": 1, "H": 2, "S": 3}
                suit_idx = suit_map.get(suit, 0)
                
                boxes.append([x_center, y_center, width, height])
                classes.append((class_id, rank_idx, suit_idx))
        boxes = torch.tensor(boxes, dtype=torch.float32)
        classes = torch.tensor(classes, dtype=torch.long)
        
        return img, boxes, classes