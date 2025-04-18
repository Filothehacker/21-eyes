from collections import defaultdict, Counter
import numpy as np
import torch


def compute_iou_numpy(box1_center, box1_size, box2_center, box2_size):
    
    box1_min = box1_center - box1_size/2
    box1_max = box1_center + box1_size/2
    box2_min = box2_center - box2_size/2
    box2_max = box2_center + box2_size/2

    # Intersection area
    inter_min = np.maximum(box1_min, box2_min)
    inter_max = np.minimum(box1_max, box2_max)
    intersection_area = np.clip(inter_max-inter_min, a_min=0, a_max=None)
    intersection_area = np.prod(intersection_area, axis=-1)

    # Union area
    box1_area = np.prod(box1_size, axis=-1)
    box2_area = np.prod(box2_size, axis=-1)
    union_area = box1_area + box2_area - intersection_area

    # IoU
    iou = intersection_area/(union_area+1e-6)
    return iou


### ---------------------------------------------------------------------------------------------------- ###
### ---------- PROCESS PREDICTIONS --------------------------------------------------------------------- ###


def process_pred(pred, B):

    # Find the predicted class and concatenate it back with the bounding boxes
    class_probs = pred[..., 5:]
    class_id = torch.argmax(class_probs, dim=-1).unsqueeze(-1)
    boxes = torch.cat([pred[...,:B*5], class_id], dim=-1)

    return boxes.detach().cpu().numpy()


def process_true(true):

    # Find the predicted class and concatenate it back with the bounding boxes
    class_probs = true[..., 5:]
    class_id = torch.argmax(class_probs, dim=-1).unsqueeze(-1)
    boxes = torch.cat([true[...,:5], class_id], dim=-1)

    return boxes.detach().cpu().numpy()


def convert_boxes_to_list_batch(boxes_batch, S, B, resize=False):

    boxes_list = []
    confidences_list = []
    classes_list = []

    for boxes in boxes_batch:
        boxes_img, confidences_img, classes_img = convert_boxes_to_list(boxes, S, B, resize)
        boxes_list.append(boxes_img)
        confidences_list.append(confidences_img)
        classes_list.append(classes_img)
    
    return boxes_list, confidences_list, classes_list


def convert_boxes_to_list(boxes, S, B, resize=False):
    
    boxes_list = []
    confidences_list = []
    classes_list = []

    for sx in range(S):
        for sy in range(S):
            for b in range(B):
                if boxes[sx, sy, b*5+4] <= 0:
                    continue

                box = boxes[sx, sy, b*5:b*5+4]
                confidence = float(boxes[sx, sy, b*5+4])
                class_id = int(boxes[sx, sy, b*5+5])

                x_center = (sx+box[0]) / S if resize else box[0]
                y_center = (sy+box[1]) / S if resize else box[1]
                width = box[2]
                height = box[3]

                boxes_list.append(np.array([x_center, y_center, width, height]))
                confidences_list.append(confidence)
                classes_list.append(class_id)

    return boxes_list, confidences_list, classes_list


### ---------------------------------------------------------------------------------------------------- ###
### ---------- NON-MAXIMUM SUPPRESSION ----------------------------------------------------------------- ###


def apply_non_max_suppression_batch(boxes_batch, confidences_batch, classes_batch, confidence_threshold=0.5, iou_threshold=0.5):

    new_boxes = []
    new_confidences = []
    new_classes = []

    for i in range(len(boxes_batch)):
        boxes_img, confidences_img, classes_img = apply_non_max_suppression(
            boxes_batch[i],
            confidences_batch[i],
            classes_batch[i],
            confidence_threshold,
            iou_threshold
        )

        new_boxes.append(boxes_img)
        new_confidences.append(confidences_img)
        new_classes.append(classes_img)

    return new_boxes, new_confidences, new_classes


def apply_non_max_suppression(boxes, confidences, classes, confidence_threshold=0.5, iou_threshold=0.5):

    if len(boxes) == 0:
        return []

    sorted_idxs = np.argsort(confidences)[::-1]
    sorted_idxs = [i for i in sorted_idxs if confidences[i] > confidence_threshold]
    boxes_sorted = [boxes[i] for i in sorted_idxs]
    confidences_sorted = [confidences[i] for i in sorted_idxs]
    classes_sorted = [classes[i] for i in sorted_idxs]
    
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
            iou = compute_iou_numpy(
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


### ---------------------------------------------------------------------------------------------------- ###
### ---------- MEAN AVERAGE PRECISION ------------------------------------------------------------------ ###


def compute_map(pred, true, S, B, num_classes=52, confidence_threshold=0.5, iou_threshold=0.5):

    # Convert the predictions and true values to lists
    processed_pred = process_pred(pred, B)
    processed_true = process_true(true)

    pred_boxes_batch, pred_confidences_batch, pred_classes_batch = convert_boxes_to_list_batch(processed_pred, S, B)
    true_boxes_batch, true_confidences_batch, true_classes_batch = convert_boxes_to_list_batch(processed_true, S, B=1)

    # Apply non-maximum suppression
    pred_boxes_batch, pred_confidences_batch, pred_classes_batch = apply_non_max_suppression_batch(
        pred_boxes_batch, pred_confidences_batch, pred_classes_batch, iou_threshold=iou_threshold, confidence_threshold=confidence_threshold
    )

    # Initialize the arrays to store the precision and recall values
    batch_size = len(pred_boxes_batch)
    precisions = -np.ones((batch_size, num_classes))
    recalls = -np.ones((batch_size, num_classes))

    for idx in range(batch_size):
        pred_boxes = pred_boxes_batch[idx]
        pred_confidences = pred_confidences_batch[idx]
        pred_classes = pred_classes_batch[idx]
        preds = defaultdict(list)

        true_boxes = true_boxes_batch[idx]
        true_confidences = true_confidences_batch[idx]
        true_classes = true_classes_batch[idx]

        true_object_idxs = [i for i, conf in enumerate(true_confidences) if conf == 1]
        true_boxes = [true_boxes[i] for i in true_object_idxs]
        true_confidences = [true_confidences[i] for i in true_object_idxs]
        true_classes = [true_classes[i] for i in true_object_idxs]

        # Count the occurrences of each class in the true labels and create a dictionary to track true matches
        true_class_dict = dict(Counter(true_classes))
        true_matces_by_class = {
            class_id: [False]*count for class_id, count in true_class_dict.items()
        }

        # Sort the predictions by confidence
        sorted_idxs = np.argsort(pred_confidences)[::-1]
        for i in sorted_idxs:
            pred_box = pred_boxes[i]
            pred_confidence = pred_confidences[i]
            pred_class = pred_classes[i]

            # Check if the predicted class is in the true classes
            true_matches_idx = [j for j, c in enumerate(true_classes) if c == pred_class]
            if not true_matches_idx:
                continue
            # Get the true boxes relative to the class
            true_boxes_same_class = np.array(true_boxes)[true_matches_idx]

            # Compute the IoU for each true box with the predicted box
            ious = []
            for box in true_boxes_same_class:
                iou = compute_iou_numpy(
                    pred_box[:2],
                    pred_box[2:4],
                    box[:2],
                    box[2:4]
                )
                ious.append(iou)

            max_iou_idx = np.argmax(ious)
            max_iou = ious[max_iou_idx]
            true_idx = true_matches_idx[max_iou_idx]

            # Check if the IoU is above the threshold and if the true box has not been matched yet
            if max_iou >= iou_threshold and not true_matces_by_class[pred_class][true_matches_idx.index(true_idx)]:
                # If it is a TP, mark it as matched and add it to the predictions
                true_matces_by_class[pred_class][true_matches_idx.index(true_idx)] = True
                preds[pred_class].append([pred_confidence, 1])
            else:
                # If it is a FP, just add it to the predictions
                preds[pred_class].append([pred_confidence, 0])
    
        # If a class has been predicted xor is in the true labels, set precision and recall to 0
        # TODO: check if this is necessary (I think it is)
        for class_id in range(num_classes):
            if class_id not in list(preds.keys()):
                if class_id in list(true_class_dict.keys()):
                    precisions[idx, class_id] = 0
                    recalls[idx, class_id] = 0
                continue
            if class_id not in list(true_class_dict.keys()):
                if class_id in list(preds.keys()):
                    precisions[idx, class_id] = 0
                    recalls[idx, class_id] = 0
                continue

            # Sort the predictions within the class by confidence
            sorted_preds_class = sorted(preds[class_id], key=lambda x: x[0], reverse=True)
            
            # Accumulate tp and fp
            tp_cumsum = 0
            fp_cumsum = 0
            for pred_score, is_tp in sorted_preds_class:
                if is_tp:
                    tp_cumsum += 1
                else:
                    fp_cumsum += 1
                    
            # Compute and store precision and recall
            precision = tp_cumsum / (tp_cumsum + fp_cumsum)
            precisions[idx, class_id] = precision
            recall = tp_cumsum / true_class_dict[class_id]
            recalls[idx, class_id] = recall

    # Initialize the average precision for each class separately
    average_precisions = []
    for class_id in range(num_classes):
        
        # Filter out the classes that are neither in the predictions nor in the true labels of the batch
        class_precisions = precisions[:,class_id]
        class_precisions = class_precisions[class_precisions != -1]
        class_recalls = recalls[:,class_id]
        class_recalls = class_recalls[class_recalls != -1]

        if len(class_precisions) == 0:
            continue
        # Insert a 0 in the arrays for the interpolation method
        class_precisions = np.insert(class_precisions, 0, 0)
        class_recalls = np.insert(class_recalls, 0, 0)

        # Calculate the average precision using the 11-point interpolation method
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(class_recalls >= t) == 0:
                p = 0
            else:
                p = np.max(class_precisions[class_recalls >= t])
            ap += p / 11
        average_precisions.append(float(ap))

    # Average the average precisions over all classes
    mean_ap = float(np.mean(average_precisions))
    return mean_ap