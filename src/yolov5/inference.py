from collections import Counter, defaultdict
import cv2
import numpy as np
import os
import torch
from torchvision import transforms
from visualize import visualize_pred
import yaml
from yolov5_ultralytics.models.yolo import Model
from yolov5_ultralytics.utils.general import non_max_suppression


def compute_iou_numpy():
    pass


def process_pred_batch(boxes):
    pass


def process_pred(boxes):

    boxes_ = boxes.detach().cpu().numpy()
    boxes = boxes_[:, :4].tolist()
    confidences = boxes_[:, 4].tolist()
    class_ids = boxes_[:, 5].astype(int).tolist()
    
    return boxes, confidences, class_ids


def process_true(labels, batch_size):

    labels = labels.detach().cpu().numpy()
    boxes = []
    class_ids = []

    idx = 0
    while idx < batch_size:
        labels_img = labels[labels[:, 0] == idx]
        boxes.append(labels_img[:, 2:].tolist())
        class_ids.append(labels_img[:, 1].astype(int).tolist())
        idx += 1
    
    return boxes, class_ids



def compute_map(pred, true, num_classes=52, conf_threshold=0.25, iou_threshold=0.45):

    pred_nms = non_max_suppression(pred, conf_thres=conf_threshold, iou_thres=iou_threshold)
    pred_boxes, pred_confidences, pred_classes = process_pred_batch(pred_nms)
    batch_size = pred[0].shape[0]
    true_boxes, true_classes = process_true(true, batch_size)

    # Initialize the arrays to store the precision and recall values
    batch_size = len(pred_boxes)
    precisions = -np.ones((batch_size, num_classes))
    recalls = -np.ones((batch_size, num_classes))

    for idx in range(batch_size):
        pred_boxes = pred_boxes[idx]
        pred_confidences = pred_confidences[idx]
        pred_classes = pred_classes[idx]
        preds = defaultdict(list)

        true_boxes = true_boxes[idx]
        true_confidences = true_confidences[idx]
        true_classes = true_classes[idx]

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


if __name__ == "__main__":
    cwd = os.getcwd()

    # Load the classes
    classes_path = os.path.join(cwd, "configurations", "classes.yaml")
    with open(classes_path, "r") as f:
        classes = yaml.safe_load(f)
    CLASSES = classes["classes"]

    # Load the image
    print("Loading an image...")
    image_path = os.path.join(cwd, "data_yolo", "test", "images", "000246247_jpg.rf.fb915aef7c063ce2ac971f8de0d8b2c1.jpg")
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transforms.ToTensor()(image)

    # Create the model
    print("Instantiating the model...")
    model_config_path = os.path.join(cwd, "src", "yolov5", "yolov5_ultralytics", "models", "yolov5s.yaml")
    yolov5 = Model(model_config_path, ch=3, nc=len(CLASSES))

    # Load the pre-trained weights for the convolution
    model_path = os.path.join(cwd, "models", "yolov5_best.pth")
    model_state = torch.load(model_path)
    yolov5.load_state_dict(model_state["model_state_dict"], strict=False)

    # Do the forward pass
    print("Passing the image through the model...")
    yolov5.eval()
    with torch.no_grad():
        output = yolov5(image.unsqueeze(0))

    # Do inference
    print("Doing inference...")
    boxes = non_max_suppression(output)[0]

    boxes, confidences, class_ids = process_pred(boxes)

    visualize_pred(
        image=image.permute(1, 2, 0).numpy(),
        boxes=boxes,
        classes_id=class_ids,
        classes=CLASSES
    )