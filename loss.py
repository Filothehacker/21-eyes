import torch
from yolo_functions import compute_iou



class YOLOv1Loss(torch.nn.Module):

    def __init__(self, yolo_params, lambda_coord=5.0, lambda_noobj=0.5, device="cpu"):
        super(YOLOv1Loss, self).__init__()

        self.B = yolo_params["B"]
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.device = device
    

    def forward(self, pred, true):
        """
        Compute the YOLO loss function.
        This is specific to the first version => check for other versions.

        :param pred: Output from the model
        :param true: Ground truth
        :return: Loss value
        """

        B = self.B
        lambda_coord = self.lambda_coord
        lambda_noobj = self.lambda_noobj

        # Get a mask for the position of the object in the grid cell
        # This is specific to YOLOv1, as a cell is responsible for predicting an object if it falls into it
        object_mask = true[..., 4].unsqueeze(-1)

        # Get indexes of the boxes responsible for each cell
        ious = torch.cat([compute_iou(
            pred[..., b*5:b*5+2],
            pred[..., b*5+2:b*5+4],
            true[..., :2],
            true[..., 2:4]
        ).unsqueeze(0) for b in range(B)], dim=0)
        best_box = torch.argmax(ious, dim=0).unsqueeze(-1)

        # Mask out the predicted coordinates of the boxes that are not responsible for the object
        # pred_box_masked = object_mask*(best_box*pred[..., 5:9] + (1-best_box)*pred[..., :4])
        pred_box_masked = sum([
            object_mask*(best_box == b) * pred[..., b*5:b*5+4]
            for b in range(B)
        ])

        # Mask out the true coordinates of the boxes that are not responsible
        true_box_masked = object_mask*true[..., :4]

        # Take the square root of the width and height of the boxes

        pred_size = torch.sign(pred_box_masked[..., 2:4]) * torch.sqrt(torch.abs(pred_box_masked[..., 2:4])+1e-6)
        true_size = torch.sqrt(true_box_masked[..., 2:4])
        
        pred_box_masked = torch.cat([pred_box_masked[..., :2], pred_size], dim=-1)
        true_box_masked = torch.cat([true_box_masked[..., :2], true_size], dim=-1)


        # 1. Compute the coordinate loss for the cells that contain an object
        coord_loss = lambda_coord * torch.sum((pred_box_masked - true_box_masked)**2, dim=[1, 2, 3])


        # 2. Compute the confidence loss for the cells that contain an object
        pred_conf_obj_masked = sum([
            object_mask*(best_box == b) * pred[..., b*5+4].unsqueeze(-1)
            for b in range(B)
        ])
        true_conf_obj_masked = object_mask*true[..., 4].unsqueeze(-1)

        obj_conf_loss = torch.sum((pred_conf_obj_masked - true_conf_obj_masked)**2, dim=[1, 2, 3])


        # 3. Compute the confidence loss for the cells that do not contain an object
        pred_conf_noobj_masked = sum([
            (1-object_mask)*(best_box == b) * pred[..., b*5+4].unsqueeze(-1)
            for b in range(B)
        ])
        true_conf_noobj_masked = (1-object_mask) * true[..., 4].unsqueeze(-1)
        noobj_conf_loss = lambda_noobj * torch.sum((pred_conf_noobj_masked - true_conf_noobj_masked)**2, dim=[1, 2, 3])

        # 4. Compute the class loss for the cells that contain an object
        class_loss = torch.sum((true[..., 5:]-pred[..., B*5:])**2, dim=-1).unsqueeze(-1)
        class_loss_masked = torch.sum((object_mask * class_loss), dim=[1, 2, 3])

        total_loss = coord_loss + obj_conf_loss + noobj_conf_loss + class_loss_masked
        return total_loss.sum()