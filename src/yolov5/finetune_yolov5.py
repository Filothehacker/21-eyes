import gc
from datasets import CustomDataset, custom_collate_fn
from dotenv import load_dotenv
import json
import os
import torch
from torch.utils.data import DataLoader
from yolov5_ultralytics.models.yolo import Model
from yolov5_ultralytics.utils.loss import ComputeLoss
from train import train, eval
import wandb
import yaml


def train_loop(model, train_loader, development_loader, criterion, optimizer, scheduler, scaler, device, num_epochs, cwd):
    
    torch.cuda.empty_cache()
    gc.collect()

    # Mount all info abou the run on wandb
    wandb.watch(model, log="all")

    # Track metrics
    best_eval_loss = float("inf")
    start_lr = optimizer.param_groups[0]["lr"]
    
    for epoch in range(num_epochs):
        print("\nEpoch {}/{}".format(epoch+1, num_epochs))

        # Retrieve the learning rate
        curr_lr = optimizer.param_groups[0]["lr"]

        # Train
        train_loss = train(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            device=device
        )
        print("Train\t Loss: {:.04f}\t Learning rate: {:.04f}".format(train_loss, curr_lr))

        # Evaluate
        eval_loss = eval(
            model=model,
            data_loader=development_loader,
            criterion=criterion,
            device=device
        )
        print("Eval\t Loss: {:.04f}".format(eval_loss))

        # Save the model if the eval map is the best so far
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            checkpoint_path = os.path.join(cwd, "models", "yolo_v5.pth")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "eval_loss": eval_loss,
                    "epoch": epoch
                },
                checkpoint_path
            )
            print("Model saved!")
            
        # Log metrics to wandb
        wandb.log({
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "lr": curr_lr
        })

        # Update the learning rate
        if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            continue

        if epoch < 10:
            for param_group in optimizer.param_groups:
                param_group["lr"] = start_lr * (epoch+1)
        
        if epoch >= 50:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(eval_loss)
            else:
                scheduler.step()

    return best_eval_loss


if __name__ == "__main__":

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", DEVICE)
    cwd = os.getcwd()

    # Load the environment variables (wandb api key)
    load_dotenv()
    api_key = os.getenv("WANDB_KEY")

    # Load the configuration file
    print("Reading the configuration files...")
    train_config_path = os.path.join(cwd, "configurations", "finetune_config.json")
    with open(train_config_path, "r") as f:
        train_config = json.load(f)
    MODEL_PARAMS = {
        "S": 7,
        "B": 2,
        "C": 52
    }

    # Load the classes
    classes_path = os.path.join(cwd, "configurations", "classes.yaml")
    with open(classes_path, "r") as f:
        classes = yaml.safe_load(f)
    CLASSES = classes["classes"]

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
        batch_size=train_config["batch_size"],
        num_workers=4 if DEVICE == "cuda" else 0,
        shuffle=True,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )

    development_data = CustomDataset(
        images_dir=os.path.join(cwd, "data_yolo", "development", "images"),
        labels_dir=os.path.join(cwd, "data_yolo", "development", "labels"),
        transform=None,
        resize=False
    )

    development_loader = DataLoader(
        dataset=development_data,
        batch_size=train_config["batch_size"],
        num_workers=4 if DEVICE == "cuda" else 0,
        shuffle=False,
        collate_fn=custom_collate_fn

    )

    # Instantiate the model
    print("Instantiating the model...")
    model_config_path = os.path.join(cwd, "src", "yolov5", "yolov5_ultralytics", "models", "yolov5s.yaml")
    yolov5 = Model(model_config_path, ch=3, nc=len(CLASSES))

    # Load the pre-trained weights for the convolution
    checkpoint_path = os.path.join(cwd, "models", "yolov5s_pretrained.pt")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model_state = checkpoint['model'].state_dict()
    filtered_state = {
        k: v for k, v in model_state.items()
        if k in yolov5.state_dict() and v.shape == yolov5.state_dict()[k].shape
    }
    yolov5.load_state_dict(filtered_state, strict=False)

    # Set the hyperparameters
    hyperparameters_path = os.path.join(cwd, "src", "yolov5", "yolov5_ultralytics", "data", "hyps", "hyp.scratch-low.yaml")
    with open(hyperparameters_path, "r") as f:
        yolov5.hyp = yaml.safe_load(f)

    yolov5 = yolov5.to(DEVICE)
    
    # Create all necessary components for the training
    criterion = ComputeLoss(yolov5)

    optimizer = torch.optim.Adam(
        params=yolov5.parameters(),
        lr=train_config["lr"],
        weight_decay=train_config["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=train_config["lr"],
        steps_per_epoch=len(train_loader),
        epochs=train_config["epochs"],
    )

    scaler = torch.amp.GradScaler()

    # Start wandb run
    print("Logging in wandb...")
    wandb.login(key=api_key)
    run = wandb.init(
        name=train_config["config_name"],
        project="21-eyes",
        config=train_config
    )

    # Save the model architecture as a string and log it to wandb
    model_architecture = str(yolov5)
    archive_path = os.path.join(cwd, "models", "yolov5_architecture.txt")
    with open(archive_path, "w") as archive_file:
        archive_file.write(model_architecture)
    artifact = wandb.Artifact("yolov5_architecture", type="model")
    artifact.add_file(archive_path)
    wandb.run.log_artifact(artifact)

    # Train the model
    print("Starting the training phase...")
    best_eval_map = train_loop(
        model=yolov5,
        train_loader=train_loader,
        development_loader=development_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=DEVICE,
        num_epochs=train_config["epochs"],
        cwd=cwd
    )
    print("Best eval loss: {:.04f}".format(best_eval_map))
    
    # Terminate the wandb run
    print("Closing the run...")
    run.finish()