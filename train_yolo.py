import gc
from dotenv import load_dotenv
import os
import torch
from torch.utils.data import DataLoader
from utils import CustomDataset, train, eval
import wandb
import yaml


def train_loop(
    model, train_loader, development_loader, criterion, optimizer, scheduler, scaler, device, num_epochs,

):
    
    torch.cuda.empty_cache()
    gc.collect()

    # Mount all info abou the run on wandb
    wandb.watch(model, log="all")

    # Track metrics
    best_eval_map = 0.0
    
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
        eval_loss, eval_map = eval(
            model=model,
            data_loader=development_loader,
            criterion=criterion,
            device=device
        )
        print("Eval\t Loss: {:.04f}\t map: {:.04f}".format(eval_loss, eval_map))

        # Save the model if the eval map is the best so far
        if eval_map > best_eval_map:
            best_eval_map = eval_map
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "eval_map": eval_map,
                    "epoch": epoch
                },
                os.path.join(cwd, "models", "yolo_v1.pth")
            )
        
        # Log metrics to wandb
        wandb.log({
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "eval_map": eval_map,
            "lr": curr_lr
        })

        # Update the learning rate with the scheduler
        scheduler.step(eval_loss)

    return best_eval_map



if __name__ == "__main__":

    # Load the environment variables (wandb api key)
    load_dotenv()
    cwd = os.getcwd()
    api_key = os.getenv("WANDB_KEY")

    # Load the configiration files
    config_path = os.path.join(cwd, "configurations", "yolo_v1.yaml")
    with open(config_path, "r") as f:
        yolo_params = yaml.safe_load(f)

    # Load the datasets
    train_data = CustomDataset(
    images_dir=os.path.join(cwd, "data", "train", "images"),
    labels_dir=os.path.join(cwd, "data", "train", "labels"),
    classes=CLASSES,
    yolo_params=YOLO_PARAMS,
    transform=None,
    input_size=(448, 448)
    )

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=CONFIG["batch_size"],
        # num_workers=1,
        shuffle=True,
        pin_memory=True
    )

    development_data = CustomDataset(
        images_dir=os.path.join(cwd, "data", "development", "images"),
        labels_dir=os.path.join(cwd, "data", "development", "labels"),
        classes=CLASSES,
        yolo_params=YOLO_PARAMS,
        transform=None,
        input_size=(448, 448)
    )

    development_loader = DataLoader(
        dataset=development_data,
        batch_size=CONFIG["batch_size"],
        num_workers=8,
        shuffle=False
    )

# TODO: Load the test dataset