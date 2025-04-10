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
    def __init__(self, images_dir, labels_dir, classes, yolo_params, transform=None, input_size=(448, 448)):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.classes = classes
        self.yolo_params = yolo_params
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
    

    def build_label(self, label_path):
        params = self.yolo_params
        label = torch.zeros((params["S"], params["S"], params["B"]*5 + len(self.classes)))

        with open(label_path, "r") as f:
            for line in f:
                values = line.strip().split()
                if len(values) != 5:
                    continue
                class_id = int(values[0])
                x_center, y_center, width, height = map(float, values[1:])

                # Determine the cell that should be responsible for the object
                grid_x = int(x_center*params["S"])
                grid_y = int(y_center*params["S"])

                # Convert to coordinates relative to the cell
                x_cell = x_center*params["S"] - grid_x
                y_cell = y_center*params["S"] - grid_y

                # Skip the cell if it has already been chosen as the center of another object
                # This is specific to YOLOv1
                if label[grid_y, grid_x, 4] != 0:
                    continue

                # Set the coordinates and dimensions of the bounding boxes
                for b in range(params["B"]):
                    label[grid_y, grid_x, b*5] = x_cell
                    label[grid_y, grid_x, b*5+1] = y_cell
                    label[grid_y, grid_x, b*5+2] = width
                    label[grid_y, grid_x, b*5+3] = height
                    label[grid_y, grid_x, b*5+4] = 1
                # Set the class label
                label[grid_y, grid_x, params["B"]*5+class_id] = 1
        return label
    

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
        label = self.build_label(label_path)
        
        return img, label


def train(
        model,
        data_loader,
        criterion,
        optimizer,
        scaler,
        device
):
    # Set the model to training mode
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(data_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        # Forward pass
        with torch.amp.autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Backward pass and optimization
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update the running loss and then empty the cache
        running_loss += loss.item()
        del images, labels, outputs, loss
        torch.cuda.empty_cache()
    
    train_loss = running_loss/len(data_loader)
    return train_loss
