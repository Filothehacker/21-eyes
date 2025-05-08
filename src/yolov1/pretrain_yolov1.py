from datasets import ClassificationDataset
from dotenv import load_dotenv
import gc
import json
import os
import torch
from torch.utils.data import DataLoader
from train import train, eval_classification
import wandb
import yaml
from yolov1 import Darknet
from torch.cuda.amp import GradScaler


def train_loop(model, train_loader, development_loader, criterion, optimizer, scheduler, scaler, device, num_epochs, cwd):
    
    torch.cuda.empty_cache()
    gc.collect()

    # Mount all info abou the run on wandb
    wandb.watch(model, log="all")

    # Track metrics
    best_eval_acc = -1.0
    
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
        eval_loss, eval_acc = eval_classification(
            model=model,
            data_loader=development_loader,
            criterion=criterion,
            device=device
        )
        print("Eval\t Loss: {:.04f}\t Acc: {:.04f}".format(eval_loss, eval_acc))

        # Save the model if the eval map is the best so far
        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            checkpoint_path = os.path.join(cwd, "models", "darknet_yolov1.pth")
            torch.save(
                {
                    "model_state_dict": model.cnn.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "eval_acc": eval_acc,
                    "epoch": epoch
                },
                checkpoint_path
            )
            print("Model saved!")
        
        # Log metrics to wandb
        wandb.log({
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "eval_acc": eval_acc,
            "lr": curr_lr
        })

        # Update the learning rate with the scheduler
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(eval_loss)
        else:
            scheduler.step()
        
    # Save the model to wandb
    artifact = wandb.Artifact("darknet_yolov1", type="model")
    artifact.add_file(checkpoint_path)
    wandb.run.log_artifact(artifact)

    return best_eval_acc


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

    # Load the training configuration file
    train_config_path = os.path.join(cwd, "configurations", "pretrain_config.json")
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
    classes=CLASSES
    )

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=train_config["batch_size"],
        num_workers=4 if DEVICE == "cuda" else 0,
        shuffle=True,
        pin_memory=True
    )

    development_data = ClassificationDataset(
        images_dir=os.path.join(cwd, "data_classification", "development", "images"),
        labels_dir=os.path.join(cwd, "data_classification", "development", "labels"),
        classes=CLASSES
    )

    development_loader = DataLoader(
        dataset=development_data,
        batch_size=train_config["batch_size"],
        num_workers=4 if DEVICE == "cuda" else 0,
        shuffle=False
    )

    # Create the model and all necessary components for the training
    print("Instantiating the model...")
    darknet = Darknet(
        model_params=MODEL_PARAMS,
        cnn_blocks=CNN_DICT,
        n_classes=len(CLASSES)
    )
    
    darknet.apply(darknet.init_weights)
    darknet = darknet.to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        params=darknet.parameters(),
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

    scaler =GradScaler('DEVICE')

    # Start wandb run
    print("Logging in wandb...")
    wandb.login(key=api_key)
    run = wandb.init(
        name=train_config["config_name"],
        project="21-eyes",
        config=train_config
    )

    # Save the model architecture as a string and log it to wandb
    model_architecture = str(darknet)
    archive_path = os.path.join(cwd, "models", "darknet_yolov1_architecture.txt")
    with open(archive_path, "w") as archive_file:
        archive_file.write(model_architecture)
    artifact = wandb.Artifact("darknet_yolov1_architecture", type="model")
    artifact.add_file(archive_path)
    wandb.run.log_artifact(artifact)

    # Train the model
    print("Starting the training phase...")
    best_eval_map = train_loop(
        model=darknet,
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
    print("Best eval acc: {:.04f}".format(best_eval_map))
    
    # Terminate the wandb run
    print("Closing the run...")
    run.finish()