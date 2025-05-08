from inference import compute_map
import torch
from tqdm import tqdm


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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
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


def eval_classification(model, data_loader, criterion, device):
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


def eval_detection(model, data_loader, criterion, device):
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

    for batch, (images, labels) in enumerate(data_loader):

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        with torch.inference_mode():
            pred = model(images)
            loss = criterion(pred, labels)

        # Sum evaluation loss and mean average precision
        eval_loss += loss.item()
        eval_map += compute_map(pred, labels, model.params["S"], model.params["B"])
        
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

    # Average the loss and map over the batches
    eval_loss /= len(data_loader)
    eval_map /= len(data_loader)
    return eval_loss, eval_map