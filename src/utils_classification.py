import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import yaml


class ClassificationDataset(Dataset):
    def __init__(self, images_dir, labels_dir, classes, model_params, transform=None, input_size=(448, 448)):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.classes = classes
        self.model_params = model_params
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
        with open(label_path, "r") as f:
            label_data = f.read().strip()
        label = torch.zeros(len(self.classes), dtype=torch.float32)
        label[self.classes.index(label_data)] = 1.0
        
        return img, label


def train(model, data_loader, criterion, optimizer, scaler, device):
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

    for batch, (images, labels) in enumerate(data_loader):

        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        # Forward pass
        with torch.amp.autocast(device):
            pred = model(images)
            loss = criterion(pred, labels)

        # Backward pass and optimization
        scaler.scale(loss).backward()
        scaler.step(optimizer)
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


def eval(model, data_loader, criterion, device):

    # Set model to evaluation mode to not compute and backpropagate gradients
    model.eval()

    # Initializing evaluation metrics
    eval_loss = 0.0
    n_correct = 0

    bar = tqdm(
        total=len(data_loader),
        dynamic_ncols=True,
        leave=False,
        position=0,
        desc="Evaluation",
        unit="batch"
    )

    for batch, (images, labels) in enumerate(data_loader):

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        with torch.inference_mode():
            pred = model(images)
            loss = criterion(pred, labels)

        # Sum evaluation loss and number of correct predictions
        eval_loss += loss.item()
        n_correct += int((pred.argmax(dim=1) == labels.argmax(dim=1)).sum().item())
        
        # Update bar info
        bar.set_postfix(
            loss = "{:.04f}".format(float(eval_loss/(batch+1))),
            acc = "{:.04f}".format(float(n_correct*100/(batch+1))),
            n_correct = n_correct,
        )
        bar.update()

        # Empty the cache
        del images, labels, pred, loss
        torch.cuda.empty_cache()
    bar.close()

    # Average the loss and acc over the batches
    eval_loss /= len(data_loader)
    eval_acc = n_correct*100/(len(data_loader)*data_loader.batch_size)
    return eval_loss, eval_acc


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
    train_config_path = os.path.join(cwd, "configurations", "train_config.json")
    with open(train_config_path, "r") as f:
        train_config = json.load(f)
    
    # Load the classes
    classes_path = os.path.join(cwd, "configurations", "classes.yaml")
    with open(classes_path, "r") as f:
        classes = yaml.safe_load(f)
    CLASSES = classes["classes"]

    # Load the datasets
    print("Loading the datasets...")
    train_data = ClassificationDataset(
    images_dir=os.path.join(cwd, "data_classification", "train", "images"),
    labels_dir=os.path.join(cwd, "data_classification", "train", "labels"),
    classes=CLASSES,
    model_params=MODEL_PARAMS,
    transform=None,
    input_size=(448, 448)
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