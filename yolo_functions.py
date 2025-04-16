import numpy as np
import torch



def compute_iou(box1_center, box1_size, box2_center, box2_size):
    
    box1_min = box1_center - box1_size/2
    box1_max = box1_center + box1_size/2
    box2_min = box2_center - box2_size/2
    box2_max = box2_center + box2_size/2

    # Intersection area
    inter_min = torch.max(box1_min, box2_min)
    inter_max = torch.min(box1_max, box2_max)
    intersection_area = torch.clamp(inter_max-inter_min, min=0)
    intersection_area = intersection_area.prod(dim=-1)

    # Union area
    box1_area = box1_size.prod(dim=-1)
    box2_area = box2_size.prod(dim=-1)
    union_area = box1_area + box2_area - intersection_area

    # IoU
    iou = intersection_area/(union_area+1e-6)
    return iou


def process_preds(preds, yolo_params):

    # Retain the best box for each cell
    best_box_idx = torch.argmax(preds[..., 4], dim=-1).unsqueeze(-1)
    best_boxes = sum([
        (best_box_idx == b) * preds[..., b*5:b*5+5]
        for b in range(yolo_params["B"])
    ])

    # Find the predicted class
    class_probs = preds[..., 5:]
    class_id = torch.argmax(class_probs, dim=-1).unsqueeze(-1)

    # Concatenate the best box with the class id
    boxes = torch.cat([best_boxes, class_id], dim=-1)

    return boxes


def convert_boxes_to_list(boxes, yolo_params):
    
    S = yolo_params["S"]
    boxes_list = []
    confidences = []
    classes = []

    for sx in range(S):
        for sy in range(S):
            box = boxes[sx, sy, :4]
            confidence = boxes[sx, sy, 4]
            class_id = int(boxes[sx, sy, 5])

            x_center = (sx+box[0]) / S
            y_center = (sy+box[1]) / S
            width = box[2]
            height = box[3]

            boxes_list.append([x_center, y_center, width, height])
            confidences.append(confidence)
            classes.append(class_id)

    return boxes_list, confidences, classes


def apply_non_max_suppression(boxes, confidences, classes, confidence_threshold=0.4, iou_threshold=0.5):

    if len(boxes) == 0:
        return []

    sorted_idxs = np.argsort(confidences)[::-1]
    boxes_sorted = [boxes[i] for i in sorted_idxs if confidences[i] > confidence_threshold]
    confidences_sorted = [confidences[i] for i in sorted_idxs if confidences[i] > confidence_threshold]
    classes_sorted = [classes[i] for i in sorted_idxs if confidences[i] > confidence_threshold]
    
    new_boxes = []
    new_confidences = []
    new_classes = []

    while(True):
        if len(boxes_sorted) == 0:
            break

        box = boxes_sorted.pop()
        new_boxes.append(box)
        new_confidences.append(confidences_sorted.pop())
        new_classes.append(classes_sorted.pop())

        for i in range(len(boxes_sorted)):
            iou = compute_iou(
                box[:2],
                box[2:4],
                boxes_sorted[i][:2],
                boxes_sorted[i][2:4]
            )
            if iou > iou_threshold:
                boxes_sorted.pop(i)
                confidences_sorted.pop(i)
                classes_sorted.pop(i)
    
    return new_boxes, new_confidences, new_classes
