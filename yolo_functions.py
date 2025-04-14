import torch


def compute_iou(box1_center, box1_size, box2_center, box2_size):
    
    box1_min = box1_center - box1_size/2
    box1_max = box1_center + box1_size/2
    box2_min = box2_center - box2_size/2
    box2_max = box2_center + box2_size/2

    # Intersection area
    inter_min = torch.max(box1_min, box2_min)
    inter_max = torch.min(box1_max, box2_max)
    inter_area = torch.clamp(inter_max - inter_min, min=0)
    inter_area = inter_area.prod(dim=-1)

    # Union area
    box1_area = box1_size.prod(dim=-1)
    box2_area = box2_size.prod(dim=-1)
    union_area = box1_area + box2_area - inter_area

    # IoU
    iou = inter_area/(union_area+1e-5)
    return iou


def compute_loss_v1(pred, true, yolo_params, lambda_coord=5.0, lambda_noobj=0.5):
    """
    Compute the YOLO loss function.
    This is specific to the first version and needs adaptation for handling multiple bounding boxes.

    :param pred: Output from the model
    :param true: Ground truth
    :param lambda_coord: Weight for the coordinate loss
    :param lambda_noobj: Weight for the no-object loss (not sure)
    :return: Loss value
    """

    B = yolo_params["B"]

    # Get a mask for the position of the object in the grid cell
    # This is specific to YOLOv1, as if an object falls into a grid cell, that cell is responsible for predicting it
    object_mask = true[..., 4].unsqueeze(-1)
    print(object_mask.shape)

    # Get the predicted box coordinates and dimensions
    # Not sure about why we only care about the first bounding box
    true_box_center = true[..., 0:2]
    true_box_size = true[..., 2:4]

    coord_loss = 0.0
    confidence_loss_obj = 0.0
    confidence_loss_noobj = 0.0

    for b in range(B):
        pred_box_center = pred[..., b*5:b*5+2]
        pred_box_size = pred[..., b*5+2:b*5+4]
        pred_confidence = pred[..., b*5+4]

        # 1. Compute the coordinate loss for the cells that contain an object
        loss_center = torch.sum((true_box_center-pred_box_center)**2, dim=-1, keepdim=True)
        loss_size = torch.sum((torch.sqrt(true_box_size) - torch.sqrt(true_box_size))**2, dim=-1, keepdim=True)
        loss_masked = object_mask * (loss_center+loss_size)
        coord_loss += lambda_coord * torch.sum(loss_masked, dim=-1, keepdim=True)

        # 2. Compute the IoU for the predicted and true boxes and then the confidence loss for the cells that contain an object
        iou = compute_iou(pred_box_center, pred_box_size, true_box_center, true_box_size)
        confidence_loss_obj += object_mask * torch.sum((iou-pred_confidence)**2, dim=-1, keepdim=True)

        # 3. Compute the confidence loss for the cells that do not contain an object
        confidence_loss_noobj += lambda_noobj * (1-object_mask) * torch.sum((pred_confidence)**2, dim=-1, keepdim=True)

    # 4. Compute the classification loss
    class_loss = object_mask * torch.sum((true[..., B*5:]-pred[..., B*5:])**2, dim=-1, keepdim=True)

    # Average the batch losses
    total_loss = torch.mean(coord_loss+confidence_loss_obj+confidence_loss_noobj+class_loss)
    return total_loss


def compute_loss_v2(pred, true, yolo_params, lambda_coord=5.0, lambda_noobj=0.5):

    S = yolo_params["S"]
    B = yolo_params["B"]

    # Get the true box coordinates and dimensions
    true_box_center = true[..., 0:2]
    true_box_size = true[..., 2:4]
    object_mask = true[..., 4]

    coord_loss = 0.0
    confidence_loss_obj = 0.0
    confidence_loss_noobj = 0.0

    box_mask = torch.zeros(S, S, B)[torch.newaxis, ...]
    ious = torch.zeros(S, S, B)[torch.newaxis, ...]
    box_losses = torch.zeros(S, S, B)[torch.newaxis, ...]
    noobj_loss = torch.zeros(S, S, B)[torch.newaxis, ...]
    class_loss = torch.zeros(S, S)[torch.newaxis, ...]

    for b in range(B):
        pred_box_center = pred[..., b*5:b*5+2]
        pred_box_size = pred[..., b*5+2:b*5+4]
        pred_confidence = pred[..., b*5+4]

        # 1. Compute the coordinate loss (center and size) for the cells that contain an object
        loss_center = torch.sum((true_box_center-pred_box_center)**2, dim=-1)
        loss_size = torch.sum((torch.sqrt(true_box_size) - torch.sqrt(torch.clamp(pred_box_size, 1e-6)))**2, dim=-1)
        box_losses[..., b] += lambda_coord * (loss_center+loss_size)

        # 2. Compute the IoU for the predicted and true boxes and then the confidence loss for the cells that contain an object
        iou = compute_iou(pred_box_center, pred_box_size, true_box_center, true_box_size)
        box_losses[..., b] += (iou-pred_confidence)**2

        # 3. Compute the confidence loss for the cells that do not contain an object
        noobj_loss[..., b] += lambda_noobj * (iou-pred_confidence)**2
        
        # Store the IoU for the current bounding box
        ious[..., b] = iou

    # Create a mask for the bounding box associated to the highest IoU
    box_mask = torch.zeros(S, S, B)[torch.newaxis, ...]
    for b in range(B):
        box_mask[..., b] = (ious == ious.max(dim=-1))

    # Mask the losses
    coord_loss = torch.sum(box_mask*box_losses, dim=[1, 2, 3])
    noobj_loss = torch.sum((1-box_mask) * noobj_loss, dim=[1, 2, 3])

    # 4. Compute the confidence loss
    class_loss = torch.sum((true[..., B*5:]-pred[..., B*5:])**2, dim=-1)
    confidence_loss = torch.sum(object_mask*class_loss, dim=[1, 2])
        
    # Average the losses
    total_loss = torch.mean(coord_loss + noobj_loss + confidence_loss)
    return total_loss
    

def apply_non_max_suppression():
    pass