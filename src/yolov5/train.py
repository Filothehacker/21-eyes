from inference import compute_map
import torch
from tqdm import tqdm


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
            pred = model(images)
            loss = criterion(pred[1], labels)[0]

        # Sum evaluation loss and mean average precision
        eval_loss += loss.item()
        eval_map += compute_map(pred, labels)
        
        # Update bar info
        bar.set_postfix(
            loss = "{:.04f}".format(float(eval_loss/(batch+1))),
            map = "{:.04f}".format(float(eval_map/(batch+1)))
        )
        bar.update()

        # Empty the cache
        del images, labels, pred, loss
        torch.cuda.empty_cache()
    bar.close()

    # Average the loss over the batches
    eval_loss /= len(data_loader)
    eval_map /= len(data_loader)
    return eval_loss, eval_map