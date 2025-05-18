import cv2
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CustomDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None, resize=False):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
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

    def __getitem__(self, idx):
        # Load the image
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.resize:
            img = cv2.resize(img, self.input_size)
        if self.transform:
            img = self.transform(img)
            img = transforms.ToTensor()(img)
        else:
            img = transforms.Compose([
                # transforms.Normalize(# TODO: Add values),
                transforms.ToTensor(),
            ])(img)

        # Load the label
        label_path = os.path.join(self.labels_dir, self.label_files[idx])
        targets = []
        with open(label_path, "r") as f:
            for line in f:
                values = line.strip().split()
                if len(values) != 5:
                    continue
                class_id = int(values[0])
                x_center, y_center, width, height = map(float, values[1:])
                targets.append([idx, class_id, x_center, y_center, width, height])
        targets = torch.tensor(targets)

        return img, targets, image_path, img.shape


def custom_collate_fn(batch):
    images, targets, paths, shapes = zip(*batch)

    # Stack images into a batch tensor and concatenate targets
    images = torch.stack(images, 0)
    batch_targets = []
    for i, t in enumerate(targets):
        if t.numel() > 0:
            t[:, 0] = i
            batch_targets.append(t)
    targets = torch.cat(batch_targets, 0) if batch_targets else torch.empty((0, 6))

    return images, targets, paths, shapes
    

if __name__ == "__main__":
    cwd = os.getcwd()

    # Load the datasets
    print("Loading the datasets...")
    train_data = CustomDataset(
    images_dir=os.path.join(cwd, "data_yolo", "train", "images"),
    labels_dir=os.path.join(cwd, "data_yolo", "train", "labels"),
    transform=None,
    resize=False
    )

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=64,
        num_workers=0,
        shuffle=True,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )

    for i, (images, targets, paths, shapes) in enumerate(train_loader):
        img = images[0]
        label = targets[targets[:, 0] == 0]

        print(f"Image shape: {img.shape}")
        print(f"Label shape: {label.shape}")
        print(f"Batch: {images.shape[0]}")
        break