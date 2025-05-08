from datasets import DetectionDataset
from dotenv import load_dotenv
import gc
import json
from loss import YOLOV1Loss
import os
import torch
from torch.utils.data import DataLoader
from train import train, eval_detection
import wandb
import yaml
from yolov1 import ConvolutionBlock, YoloV1


def train_loop(model, train_loader, development_loader, criterion, optimizer, scheduler, scaler, device, num_epochs, cwd):
    
    torch.cuda.empty_cache()
    gc.collect()

    # Mount all info abou the run on wandb
    wandb.watch(model, log="all")

    # Track metrics
    best_eval_map = -1.0
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
            device=device
        )
        print("Train\t Loss: {:.04f}\t Learning rate: {:.04f}".format(train_loss, curr_lr))

        # Evaluate
        eval_loss, eval_map = eval_detection(
            model=model,
            data_loader=development_loader,
            criterion=criterion,
            device=device
        )
        print("Eval\t Loss: {:.04f}\t map: {:.04f}".format(eval_loss, eval_map))

        # Save the model if the eval map is the best so far
        if eval_map > best_eval_map:
            best_eval_map = eval_map
            checkpoint_path = os.path.join(cwd, "models", "yolov1.pth")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "eval_map": eval_map,
                    "epoch": epoch
                },
                checkpoint_path
            )
            print("Model saved!")

        # Log metrics to wandb
        wandb.log({
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "eval_map": eval_map,
            "lr": curr_lr
        })

        # Update the learning rate
        if epoch < 10:
            for param_group in optimizer.param_groups:
                param_group["lr"] = start_lr * (epoch+1)
        
        if epoch >= 50:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(eval_loss)
            else:
                scheduler.step()
            
    # Save the model to wandb
    artifact = wandb.Artifact("yolov1", type="model")
    artifact.add_file(checkpoint_path)
    wandb.run.log_artifact(artifact)

    return best_eval_map


if __name__ == "__main__":

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", DEVICE)
    cwd = os.getcwd()

    # Load the environment variables (wandb api key)
    load_dotenv()
    api_key = os.getenv("WANDB_KEY")

    # Load the model configuration file
    print("Reading the configuration files...")
    model_config_path = os.path.join(cwd, "configurations", "yolov1.json")
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
    train_data = DetectionDataset(
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
        num_workers=4 if DEVICE == "cuda" else 0,
        shuffle=True,
        pin_memory=True
    )

    development_data = DetectionDataset(
        images_dir=os.path.join(cwd, "data_yolo", "development", "images"),
        labels_dir=os.path.join(cwd, "data_yolo", "development", "labels"),
        classes=CLASSES,
        model_params=MODEL_PARAMS,
        transform=None,
        resize=True
    )

    development_loader = DataLoader(
        dataset=development_data,
        batch_size=train_config["batch_size"],
        num_workers=4 if DEVICE == "cuda" else 0,
        shuffle=False
    )

    # Create the model and all necessary components for the training
    print("Instantiating the model...")
    yolov1 = YoloV1(
        model_params=MODEL_PARAMS,
        cnn_blocks=CNN_DICT,
        mlp_dict=MLP_DICT
    ).to(DEVICE)

    # Load the weights for the convolutional layers
    print("Loading the backbone...")
    checkpoint_path = os.path.join(cwd, "models", "darknet_yolov1.pth")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model_weights = checkpoint["model_state_dict"]

    darknet = torch.nn.ModuleList(
            [
                ConvolutionBlock(
                    in_c=block["in_c"],
                    channels=block["channels"],
                    kernels=block["kernels"],
                    strides=block["strides"],
                    pool=block["pool"]
                ) for block in CNN_DICT
            ]
        ).to(DEVICE)
    darknet.load_state_dict(model_weights, strict=False)
    yolov1.cnn = darknet

    criterion = YOLOV1Loss(
        B=MODEL_PARAMS["B"],
        device=DEVICE
    )

    optimizer = torch.optim.SGD(
        params=yolov1.parameters(),
        lr=train_config["lr"],
        momentum=train_config["momentum"],
        weight_decay=train_config["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        threshold=0.1,
        factor=0.5,
        patience=5,
    )

    scaler = torch.amp.GradScaler(DEVICE)

    # Start wandb run
    print("Logging in wandb...")
    wandb.login(key=api_key)
    run = wandb.init(
        name=train_config["config_name"],
        project="21-eyes",
        config=train_config
    )

    # Save the model architecture as a string and log it to wandb
    model_architecture = str(yolov1)
    archive_path = os.path.join(cwd, "models", "yolov1_architecture.txt")
    with open(archive_path, "w") as archive_file:
        archive_file.write(model_architecture)
    artifact = wandb.Artifact("yolov1_architecture", type="model")
    artifact.add_file(archive_path)
    wandb.run.log_artifact(artifact)

    # Train the model
    print("Starting the training phase...")
    best_eval_map = train_loop(
        model=yolov1,
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
    print("Best eval map: {:.04f}".format(best_eval_map))
    
    # Terminate the wandb run
    print("Closing the run...")
    run.finish()