import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import yaml
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score


# Load YAML file
yaml_path = '/Users/filippofocaccia/Desktop/card_detector/absolute_data.yaml'
with open(yaml_path, 'r') as file:
    data_config = yaml.safe_load(file)

# Extract information from the YAML file
class_names = data_config['names']  # List of class names
num_classes = data_config['nc']     # Number of classes
train_images_dir = data_config['train']  # Path to training images
val_images_dir = data_config['val']      # Path to validation images
test_images_dir = data_config['test']    # Path to test images

# Define paths to labels directories - assuming they are in 'labels' subdirectory
train_labels_dir = train_images_dir.replace('images', 'labels')
val_labels_dir = val_images_dir.replace('images', 'labels')
test_labels_dir = test_images_dir.replace('images', 'labels')


class ConvBlock(nn.Module):
    """Basic convolutional block with BatchNorm and activation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super().__init__()
        if padding is None:
            # Same padding
            padding = kernel_size // 2
            
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                             stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1, inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class CardCornerDetector(nn.Module):
    """
    Card corner detection architecture for 416x416 input images
    Detects card corners and classifies their rank and suit
    """
    def __init__(self, num_classes=52, num_ranks=13, num_suits=4, anchors_per_scale=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_ranks = num_ranks
        self.num_suits = num_suits
        self.anchors_per_scale = anchors_per_scale
        
        # Define anchors for each detection scale
        # These would be determined based on your dataset
        # Format: [width, height] normalized by feature map cell size
        self.anchors = {
            '1/8': torch.tensor([
                [0.5, 0.5],   # Small square anchor
                [0.7, 0.7],   # Medium square anchor
                [0.9, 0.9]    # Large square anchor
            ]),
            '1/16': torch.tensor([
                [1.0, 1.0],   # Small square anchor at this scale
                [1.4, 1.4],   # Medium square anchor at this scale
                [1.8, 1.8]    # Large square anchor at this scale
            ])
        }
        
        # Backbone network (feature extraction)
        self.backbone = nn.ModuleList([
            # Input: 416x416x3
            ConvBlock(3, 32, kernel_size=3, stride=1),        # 416x416x32
            ConvBlock(32, 64, kernel_size=3, stride=2),       # 208x208x64
            
            ConvBlock(64, 64, kernel_size=1, stride=1),       # 208x208x64
            ConvBlock(64, 128, kernel_size=3, stride=2),      # 104x104x128
            
            ConvBlock(128, 128, kernel_size=1, stride=1),     # 104x104x128
            ConvBlock(128, 256, kernel_size=3, stride=2),     # 52x52x256 (1/8 scale)
            
            ConvBlock(256, 256, kernel_size=1, stride=1),     # 52x52x256
            ConvBlock(256, 512, kernel_size=3, stride=2),     # 26x26x512 (1/16 scale)
        ])
        
        # Feature pyramid to extract information at different scales
        self.feature_1_8 = nn.Sequential(
            ConvBlock(256, 256, kernel_size=1),               # 52x52x256
            ConvBlock(256, 512, kernel_size=3),               # 52x52x512
            ConvBlock(512, 256, kernel_size=1),               # 52x52x256
            ConvBlock(256, 512, kernel_size=3),               # 52x52x512
            ConvBlock(512, 256, kernel_size=1)                # 52x52x256
        )
        
        self.feature_1_16 = nn.Sequential(
            ConvBlock(512, 512, kernel_size=1),               # 26x26x512
            ConvBlock(512, 1024, kernel_size=3),              # 26x26x1024
            ConvBlock(1024, 512, kernel_size=1),              # 26x26x512
            ConvBlock(512, 1024, kernel_size=3),              # 26x26x1024
            ConvBlock(1024, 512, kernel_size=1)               # 26x26x512
        )
        
        # Detection heads for each scale
        self.heads = nn.ModuleDict()
        
        # 1/8 scale detection head (52x52 feature map)
        self._create_head('1/8', 256, self.anchors_per_scale)
            
        # 1/16 scale detection head (26x26 feature map)
        self._create_head('1/16', 512, self.anchors_per_scale)
        
    def _create_head(self, name, in_channels, num_anchors):
        """Create detection heads for a specific feature scale"""
        self.heads[f"{name}_conv"] = ConvBlock(in_channels, in_channels*2, kernel_size=3)
        
        # Detection head outputs:
        # - 5 values for each anchor: objectness + bounding box (x, y, w, h)
        # - num_ranks values for card rank classification (A, 2-10, J, Q, K)
        # - num_suits values for card suit classification (spades, hearts, diamonds, clubs)
        self.heads[f"{name}_det"] = nn.Conv2d(
            in_channels*2, 
            num_anchors * (5 + self.num_ranks + self.num_suits), 
            kernel_size=1
        )
    
    def forward(self, x):
        """
        Forward pass through the network
        Args:
            x: Input tensor of shape (batch_size, 3, 416, 416)
        Returns:
            List of detection tensors at different scales
        """
        outputs = []
        route_connections = []
        
        # Backbone feature extraction
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            
            # Save outputs that will be used for feature pyramid
            if i == 5:  # 52x52x256 (1/8 scale)
                route_connections.append(x)
            elif i == 7:  # 26x26x512 (1/16 scale)
                route_connections.append(x)
        
        # Process feature maps
        # 1/8 scale feature map (52x52)
        feature_map_1_8 = self.feature_1_8(route_connections[0])
        detection_1_8 = self.heads['1/8_conv'](feature_map_1_8)
        output_1_8 = self.heads['1/8_det'](detection_1_8)
        
        # 1/16 scale feature map (26x26)
        feature_map_1_16 = self.feature_1_16(route_connections[1])
        detection_1_16 = self.heads['1/16_conv'](feature_map_1_16)
        output_1_16 = self.heads['1/16_det'](detection_1_16)
        
        # Process raw outputs into a structured format
        outputs.append(self._process_output(output_1_8, '1/8'))
        outputs.append(self._process_output(output_1_16, '1/16'))
        
        if self.training:
            return outputs
        else:
            return self._postprocess(outputs)
    
    def _process_output(self, output, scale_name):
        """
        Process raw output tensor into structured prediction format
        Args:
            output: Raw output tensor from detection head
            scale_name: Name of the scale ('1/8' or '1/16')
        Returns:
            Dictionary with structured predictions
        """
        batch_size = output.size(0)
        
        # Get grid size based on scale
        if scale_name == '1/8':
            grid_size = 52
        elif scale_name == '1/16':
            grid_size = 26
        else:
            raise ValueError(f"Unknown scale: {scale_name}")
            
        # Number of attributes per anchor:
        # - 5 values: objectness + bounding box (x, y, w, h)
        # - num_ranks values: rank probabilities
        # - num_suits values: suit probabilities
        num_attrs = 5 + self.num_ranks + self.num_suits
        
        # Reshape output tensor from [batch, anchors * attrs, grid, grid] to [batch, grid, grid, anchors, attrs]
        output = output.view(batch_size, self.anchors_per_scale, num_attrs, grid_size, grid_size)
        output = output.permute(0, 3, 4, 1, 2).contiguous()
        
        # Create prediction structure
        pred = {}
        
        # Objectness scores (sigmoid for 0-1 probability)
        pred['objectness'] = torch.sigmoid(output[..., 0:1])
        
        # Bounding box coordinates (x, y, w, h)
        pred['box_xy'] = torch.sigmoid(output[..., 1:3])  # Center coordinates (relative to grid cell)
        pred['box_wh'] = torch.exp(output[..., 3:5]) * self.anchors[scale_name].view(1, 1, 1, self.anchors_per_scale, 2)
        
        # Generate grid cell coordinates
        grid_y, grid_x = torch.meshgrid([torch.arange(grid_size), torch.arange(grid_size)], indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=2).view(1, grid_size, grid_size, 1, 2).float()
        grid = grid.to(output.device)
        
        # Add grid cell offsets to xy predictions
        pred['box_xy'] = (pred['box_xy'] + grid) / grid_size
        pred['box_wh'] = pred['box_wh'] / grid_size
        
        # Card rank probabilities (softmax across rank dimension)
        rank_offset = 5
        rank_logits = output[..., rank_offset:rank_offset+self.num_ranks]
        pred['ranks'] = F.softmax(rank_logits, dim=-1)
        
        # Card suit probabilities (softmax across suit dimension)
        suit_offset = rank_offset + self.num_ranks
        suit_logits = output[..., suit_offset:suit_offset+self.num_suits]
        pred['suits'] = F.softmax(suit_logits, dim=-1)
        
        # Raw predictions (for loss calculation during training)
        pred['raw'] = output
        
        return pred
    
    def _postprocess(self, outputs):
        """
        Post-process the outputs for inference (NMS, combine predictions, etc.)
        Args:
            outputs: List of outputs from different scales
        Returns:
            Final predictions with NMS applied
        """
        batch_size = outputs[0]['objectness'].size(0)
        device = outputs[0]['objectness'].device
        
        # Combine predictions from all scales
        all_boxes = []
        all_scores = []
        all_ranks = []
        all_suits = []
        
        for scale_preds in outputs:
            # Reshape predictions to [batch, -1, dim]
            objectness = scale_preds['objectness'].view(batch_size, -1, 1)
            box_xy = scale_preds['box_xy'].view(batch_size, -1, 2)
            box_wh = scale_preds['box_wh'].view(batch_size, -1, 2)
            ranks = scale_preds['ranks'].view(batch_size, -1, self.num_ranks)
            suits = scale_preds['suits'].view(batch_size, -1, self.num_suits)
            
            # Convert center coordinates to corner coordinates
            boxes = torch.cat([
                box_xy - box_wh / 2,  # top-left
                box_xy + box_wh / 2   # bottom-right
            ], dim=-1)
            
            all_boxes.append(boxes)
            all_scores.append(objectness)
            all_ranks.append(ranks)
            all_suits.append(suits)
        
        # Concatenate all predictions
        boxes = torch.cat(all_boxes, dim=1)
        scores = torch.cat(all_scores, dim=1)
        ranks = torch.cat(all_ranks, dim=1)
        suits = torch.cat(all_suits, dim=1)
        
        # Apply NMS and get final predictions
        final_predictions = []
        
        for b in range(batch_size):
            # Get predictions for this batch
            batch_boxes = boxes[b]
            batch_scores = scores[b].squeeze(-1)
            batch_ranks = ranks[b]
            batch_suits = suits[b]
            
            # Filter by confidence threshold
            conf_mask = batch_scores > 0.5
            boxes_filtered = batch_boxes[conf_mask]
            scores_filtered = batch_scores[conf_mask]
            ranks_filtered = batch_ranks[conf_mask]
            suits_filtered = batch_suits[conf_mask]
            
            if len(scores_filtered) == 0:
                final_predictions.append({
                    'boxes': torch.zeros((0, 4), device=device),
                    'scores': torch.zeros(0, device=device),
                    'ranks': torch.zeros((0, self.num_ranks), device=device),
                    'suits': torch.zeros((0, self.num_suits), device=device)
                })
                continue
            
            # Get rank and suit indices
            rank_indices = torch.argmax(ranks_filtered, dim=1)
            suit_indices = torch.argmax(suits_filtered, dim=1)
            
            # Apply NMS
            keep_indices = self._nms(
                boxes_filtered, 
                scores_filtered, 
                iou_threshold=0.45
            )
            
            final_predictions.append({
                'boxes': boxes_filtered[keep_indices],
                'scores': scores_filtered[keep_indices],
                'ranks': ranks_filtered[keep_indices],
                'rank_indices': rank_indices[keep_indices],
                'suits': suits_filtered[keep_indices],
                'suit_indices': suit_indices[keep_indices]
            })
            
        return final_predictions
    
    def _nms(self, boxes, scores, iou_threshold=0.45):
        """
        Non-maximum suppression
        Args:
            boxes: Bounding boxes [N, 4] (x1, y1, x2, y2)
            scores: Confidence scores [N]
            iou_threshold: IoU threshold for considering a box as overlapping
        Returns:
            Indices of boxes to keep
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort(descending=True)
        
        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.item())
                break
                
            i = order[0].item()
            keep.append(i)
            
            # Compute IoU of the current box with remaining boxes
            xx1 = torch.max(x1[i], x1[order[1:]])
            yy1 = torch.max(y1[i], y1[order[1:]])
            xx2 = torch.min(x2[i], x2[order[1:]])
            yy2 = torch.min(y2[i], y2[order[1:]])
            
            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            intersection = w * h
            
            # IoU = intersection / union
            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / union
            
            # Keep boxes with IoU <= threshold
            mask = iou <= iou_threshold
            order = order[1:][mask]
            
        return torch.tensor(keep, device=boxes.device)


class CardCornerLoss(nn.Module):
    """
    Loss function for card corner detection and classification
    """
    def __init__(self, lambda_obj=1.0, lambda_noobj=0.5, lambda_coord=5.0, 
                lambda_rank=1.0, lambda_suit=1.0):
        super().__init__()
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_coord = lambda_coord
        self.lambda_rank = lambda_rank
        self.lambda_suit = lambda_suit
        
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCELoss(reduction='none')
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, predictions, targets):
        """
        Calculate loss
        Args:
            predictions: List of prediction dictionaries from the model
            targets: Dictionary with ground truth information
                - obj_mask: [batch, scale_idx, grid_y, grid_x, anchor_idx]
                - noobj_mask: [batch, scale_idx, grid_y, grid_x, anchor_idx]
                - box_target: [batch, scale_idx, grid_y, grid_x, anchor_idx, 4]
                - rank_target: [batch, scale_idx, grid_y, grid_x, anchor_idx]
                - suit_target: [batch, scale_idx, grid_y, grid_x, anchor_idx]
        Returns:
            Total loss and component losses
        """
        loss_components = {
            'obj': 0.0,
            'noobj': 0.0,
            'coord': 0.0,
            'rank': 0.0,
            'suit': 0.0
        }
        
        # Calculate loss for each scale
        for scale_idx, preds in enumerate(predictions):
            # Extract targets for this scale
            obj_mask = targets['obj_mask'][:, scale_idx]
            noobj_mask = targets['noobj_mask'][:, scale_idx]
            box_target = targets['box_target'][:, scale_idx]
            rank_target = targets['rank_target'][:, scale_idx]
            suit_target = targets['suit_target'][:, scale_idx]
            
            # Raw predictions
            raw_preds = preds['raw']
            
            # Objectness loss
            obj_preds = torch.sigmoid(raw_preds[..., 0])
            obj_loss = self.bce_loss(obj_preds, obj_mask.float())
            loss_components['obj'] += self.lambda_obj * obj_loss[obj_mask > 0].mean()
            loss_components['noobj'] += self.lambda_noobj * obj_loss[noobj_mask > 0].mean()
            
            # Only calculate coordinate and classification losses for positive anchors
            if obj_mask.sum() > 0:
                # Bounding box coordinate loss
                xy_preds = torch.sigmoid(raw_preds[..., 1:3])
                wh_preds = raw_preds[..., 3:5]  # raw, no sigmoid
                
                xy_target = box_target[..., 0:2]
                wh_target = box_target[..., 2:4]
                
                xy_loss = self.mse_loss(xy_preds[obj_mask > 0], xy_target[obj_mask > 0])
                wh_loss = self.mse_loss(wh_preds[obj_mask > 0], wh_target[obj_mask > 0])
                
                loss_components['coord'] += self.lambda_coord * (xy_loss.mean() + wh_loss.mean())
                
                # Card rank classification loss
                rank_offset = 5
                rank_logits = raw_preds[..., rank_offset:rank_offset+13]
                rank_loss = self.ce_loss(
                    rank_logits[obj_mask > 0], 
                    rank_target[obj_mask > 0].long()
                )
                loss_components['rank'] += self.lambda_rank * rank_loss.mean()
                
                # Card suit classification loss
                suit_offset = rank_offset + 13
                suit_logits = raw_preds[..., suit_offset:suit_offset+4]
                suit_loss = self.ce_loss(
                    suit_logits[obj_mask > 0], 
                    suit_target[obj_mask > 0].long()
                )
                loss_components['suit'] += self.lambda_suit * suit_loss.mean()
        
        # Calculate total loss
        total_loss = sum(loss_components.values())
        
        return total_loss, loss_components


class CustomDataset(Dataset):
    def __init__(self, images_dir, labels_dir, class_names, transform=None, input_size=416):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.class_names = class_names  # List of class names for mapping
        self.transform = transform
        self.input_size = input_size
        
        # Get all valid image files
        self.image_files = []
        self.label_files = []
        
        for img_file in sorted(os.listdir(images_dir)):
            if not img_file.endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            label_file = os.path.splitext(img_file)[0] + '.txt'
            if not os.path.exists(os.path.join(labels_dir, label_file)):
                continue
                
            self.image_files.append(img_file)
            self.label_files.append(label_file)
                
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        
        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
        else:
            # Basic transformation if none provided
            image = transforms.Compose([
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])(image)

        # Load corresponding label
        label_path = os.path.join(self.labels_dir, self.label_files[idx])
        
        boxes = []
        classes = []
        
        with open(label_path, 'r') as f:
            for line in f:
                values = line.strip().split()
                if len(values) != 5:  # YOLO format: class_id, x_center, y_center, width, height
                    continue
                    
                class_id, x_center, y_center, width, height = map(float, values)
                class_id = int(class_id)
                
                # Extract rank and suit from class_id or class name
                class_name = self.class_names[class_id]
                
                # Extract rank (first character(s)) and suit (last character)
                if class_name.startswith('10'):
                    rank = '10'
                    suit = class_name[2]
                else:
                    rank = class_name[0]
                    suit = class_name[1]
                
                # Map rank to index (0-12)
                rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, 
                           '9': 7, '10': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
                rank_idx = rank_map.get(rank, 0)
                
                # Map suit to index (0-3)
                suit_map = {'C': 0, 'D': 1, 'H': 2, 'S': 3}
                suit_idx = suit_map.get(suit, 0)
                
                # Store bounding box in [x_center, y_center, width, height] format (normalized)
                boxes.append([x_center, y_center, width, height])
                classes.append((class_id, rank_idx, suit_idx))
        
        # Convert to tensor
        boxes = torch.tensor(boxes, dtype=torch.float32)
        classes = torch.tensor(classes, dtype=torch.long)
        
        return image, boxes, classes
        

def build_targets(batch_boxes, batch_classes, model, num_scales=2):
    """
    Convert ground truth boxes and classes to the target format expected by the loss function
    
    Args:
        batch_boxes: List of tensors with shape [num_boxes, 4] (x_center, y_center, width, height)
        batch_classes: List of tensors with shape [num_boxes, 3] (class_id, rank_idx, suit_idx)
        model: The detection model
        num_scales: Number of detection scales
        
    Returns:
        Dictionary with target tensors:
            - obj_mask: [batch, scale_idx, grid_y, grid_x, anchor_idx]
            - noobj_mask: [batch, scale_idx, grid_y, grid_x, anchor_idx]
            - box_target: [batch, scale_idx, grid_y, grid_x, anchor_idx, 4]
            - rank_target: [batch, scale_idx, grid_y, grid_x, anchor_idx]
            - suit_target: [batch, scale_idx, grid_y, grid_x, anchor_idx]
    """
    batch_size = len(batch_boxes)
    device = next(model.parameters()).device
    
    # Grid sizes for each scale
    grid_sizes = [52, 26]  # For 1/8 and 1/16 scales with 416x416 input
    
    # Initialize target tensors
    obj_mask = torch.zeros(batch_size, num_scales, max(grid_sizes), max(grid_sizes), 
                           model.anchors_per_scale, device=device)
    noobj_mask = torch.ones(batch_size, num_scales, max(grid_sizes), max(grid_sizes), 
                           model.anchors_per_scale, device=device)
    box_target = torch.zeros(batch_size, num_scales, max(grid_sizes), max(grid_sizes), 
                           model.anchors_per_scale, 4, device=device)
    rank_target = torch.zeros(batch_size, num_scales, max(grid_sizes), max(grid_sizes), 
                           model.anchors_per_scale, device=device)
    suit_target = torch.zeros(batch_size, num_scales, max(grid_sizes), max(grid_sizes), 
                           model.anchors_per_scale, device=device)
    
    # Get anchors for each scale
    anchors = []
    for scale_idx, scale_name in enumerate(['1/8', '1/16']):
        anchors.append(model.anchors[scale_name])
    
    # For each image in the batch
    for batch_idx in range(batch_size):
        boxes = batch_boxes[batch_idx]
        classes = batch_classes[batch_idx]
        
        if len(boxes) == 0:
            continue
            
        # For each scale
        for scale_idx, grid_size in enumerate(grid_sizes):
            # Get anchors for this scale
            scale_anchors = anchors[scale_idx]
            
            # For each ground truth box
            for box_idx, (box, cls) in enumerate(zip(boxes, classes)):
                # Extract class info
                class_id, rank_idx, suit_idx = cls
                
                # Convert normalized box coordinates to grid coordinates
                x_center, y_center, width, height = box
                x_center_grid = x_center * grid_size
                y_center_grid = y_center * grid_size
                width_grid = width * grid_size
                height_grid = height * grid_size
                
                # Get grid cell indices
                grid_x = int(x_center_grid)
                grid_y = int(y_center_grid)
                
                # Ensure indices are within bounds
                if grid_x >= grid_size or grid_y >= grid_size:
                    continue
                    
                # Find best matching anchor for this box
                box_wh = torch.tensor([width, height], device=device)
                anchor_wh = scale_anchors
                
                # Calculate IoU between box and anchors
                intersect_wh = torch.min(box_wh, anchor_wh)
                intersect_area = intersect_wh[:, 0] * intersect_wh[:, 1]
                box_area = width * height
                anchor_area = anchor_wh[:, 0] * anchor_wh[:, 1]
                union_area = box_area + anchor_area - intersect_area
                iou = intersect_area / union_area
                
                # Find best anchor
                best_anchor_idx = torch.argmax(iou)
                
                # Set object mask and clear noobj mask
                obj_mask[batch_idx, scale_idx, grid_y, grid_x, best_anchor_idx] = 1
                noobj_mask[batch_idx, scale_idx, grid_y, grid_x, best_anchor_idx] = 0