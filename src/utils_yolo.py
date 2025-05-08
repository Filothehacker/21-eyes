from inference import compute_map
import numpy as np
import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import yaml


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


class YoloDataset(Dataset):
    def __init__(self, images_dir, labels_dir, classes, model_params, transform=None, resize=True):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.classes = classes
        self.model_params = model_params
        self.transform = transform
        self.resize = resize
        self.input_size = (448, 448)
        
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
        params = self.model_params
        label = torch.zeros((params["S"], params["S"], 5 + len(self.classes)))

        with open(label_path, "r") as f:
            for line in f:
                values = line.strip().split()
                if len(values) != 5:
                    continue
                class_id = int(values[0])
                x_center, y_center, width, height = map(float, values[1:])

                # Determine the cell responsible for the object
                grid_x = int(x_center*params["S"])
                grid_y = int(y_center*params["S"])

                # Convert to relative coordinates
                x_cell = x_center*params["S"] - grid_x
                y_cell = y_center*params["S"] - grid_y

                # Double check
                # x_center_ = x_center*self.input_size[0]
                # y_center_ = y_center*self.input_size[0]
                # n_pixels_per_cell = self.input_size[0] / params["S"]
                # x_cell_ = (x_center_ % (n_pixels_per_cell)) / n_pixels_per_cell
                # y_cell_ = (y_center_ % (n_pixels_per_cell)) / n_pixels_per_cell
                # assert abs(x_cell - x_cell_) < 1e-6

                # Skip the cell if it has already been chosen as the center of another object
                # This is specific to YOLOv1
                if label[grid_y, grid_x, 4] != 0:
                    continue

                # Set the coordinates and dimensions of the bounding boxes
                label[grid_y, grid_x, 0] = x_cell
                label[grid_y, grid_x, 1] = y_cell
                label[grid_y, grid_x, 2] = width
                label[grid_y, grid_x, 3] = height
                label[grid_y, grid_x, 4] = 1
                # Set the class label
                label[grid_y, grid_x, 5+class_id] = 1
        return label
    

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        img = Image.open(img_path).convert("RGB")
        
        if self.resize:
            img = img.resize(self.input_size)
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.Compose([
                # transforms.Resize((448, 448)) if self.resize else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
            ])(img)

        label_path = os.path.join(self.labels_dir, self.label_files[idx])
        label = self.build_label(label_path)
        
        return img, label


def train(model, data_loader, criterion, optimizer, scaler, scheduler=None, device="cpu"):
    # Set the model to training mode
    model.train()
    train_loss = 0.0

    bar = tqdm(
        total=len(data_loader),
        dynamic_ncols=True,
        leave=False,
        position=0,
        desc="Training",
        unit="batch"
    )

    for batch, (images, labels, _, _) in enumerate(data_loader):

        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        # Forward pass
        with torch.amp.autocast(device):
            pred = model(images)
            loss = criterion(pred, labels)[0]

        # Backward pass and optimization
        scaler.scale(loss).backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        if scheduler is not None:
            scheduler.step()
        scaler.update()
        train_loss += loss.item()

        # Update bar info 
        bar.set_postfix(
            loss="{:.04f}".format(float(train_loss/(batch+1))),
            lr="{:.04f}".format(float(optimizer.param_groups[0]["lr"]))
        )
        bar.update()

        # Empty the cache
        del images, labels, pred, loss
        torch.cuda.empty_cache()
    bar.close()
    
    # Average the loss over the batches
    train_loss /= len(data_loader)
    return train_loss


def eval(model, data_loader, criterion, device="cpu"):

    # Set model to evaluation mode to not compute and backpropagate gradients
    model.eval()

    # Initializing evaluation metrics
    eval_loss = 0.0
    eval_map = 0.0
    bar = tqdm(
        total=len(data_loader),
        dynamic_ncols=True,
        leave=False,
        position=0,
        desc="Evaluation",
        unit="batch"
    )

    for batch, (images, labels, _, _) in enumerate(data_loader):

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        with torch.inference_mode():
            pred = model(images)[1]
            loss = criterion(pred, labels)[0]

        # Sum evaluation loss and mean average precision
        eval_loss += loss.item()
        # eval_map += compute_map(pred, labels, model.params["S"], model.params["B"])
        
        # Update bar info
        bar.set_postfix(
            loss = "{:.04f}".format(float(eval_loss/(batch+1))),
            # map = "{:.04f}".format(float(eval_map/(batch+1)))
        )
        bar.update()

        # Empty the cache
        del images, labels, pred, loss
        torch.cuda.empty_cache()
    bar.close()

    # Average the loss and map over the batches
    eval_loss /= len(data_loader)
    # eval_map /= len(data_loader)
    return eval_loss, eval_map


if __name__ == "__main__":
    cwd = os.getcwd()

    # Load the model configuration file
    print("Reading the configuration files...")
    model_config_path = os.path.join(cwd, "configurations", "yolo_v1.json")
    with open(model_config_path, "r") as f:
        model_config = json.load(f)

    # Retrieve the parameters
    MODEL_PARAMS = model_config["MODEL_PARAMS"]
    CNN_DICT = model_config["CNN"]
    MLP_DICT = model_config["MLP"]
    OUTPUT_SIZE = MODEL_PARAMS["S"]*MODEL_PARAMS["S"] * (MODEL_PARAMS["B"]*5+MODEL_PARAMS["C"])
    MLP_DICT["out_size"] = OUTPUT_SIZE

    # Load the training configuration file
    train_config_path = os.path.join(cwd, "configurations", "finetune_config.json")
    with open(train_config_path, "r") as f:
        train_config = json.load(f)
    
    # Load the classes
    classes_path = os.path.join(cwd, "configurations", "classes.yaml")
    with open(classes_path, "r") as f:
        classes = yaml.safe_load(f)
    CLASSES = classes["classes"]

    # Load the datasets
    print("Loading the datasets...")
    train_data = YoloDataset(
    images_dir=os.path.join(cwd, "data_yolo", "train", "images"),
    labels_dir=os.path.join(cwd, "data_yolo", "train", "labels"),
    classes=CLASSES,
    model_params=MODEL_PARAMS,
    transform=None,
    resize=True
    )

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=train_config["batch_size"],
        num_workers=0,
        shuffle=True,
        pin_memory=True
    )

    for i, (batch, batch_label) in enumerate(train_loader):
        img = batch[0]
        label = batch_label[0]
        print(f"Image shape: {img.shape}")
        print(f"Label shape: {label.shape}")
        print(f"Batch: {batch.shape[0]}")
        break