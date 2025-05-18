import cv2
from inference import compute_map, process_pred
import os
import torch
from torchvision import transforms
from visualize import visualize_pred
import yaml
from yolov5_ultralytics.models.yolo import Model
from yolov5_ultralytics.utils.general import non_max_suppression


if __name__ == "__main__":
    cwd = os.getcwd()

    # Load the classes
    classes_path = os.path.join(cwd, "configurations", "classes.yaml")
    with open(classes_path, "r") as f:
        classes = yaml.safe_load(f)
    CLASSES = classes["classes"]

    # Load the model
    print("Instantiating the model...")
    model_config_path = os.path.join(cwd, "src", "yolov5", "yolov5_ultralytics", "models", "yolov5s.yaml")
    yolov5 = Model(model_config_path, ch=3, nc=len(CLASSES))

    model_path = os.path.join(cwd, "models", "yolov5_best.pth")
    model_state = torch.load(model_path, map_location="cpu")
    yolov5.load_state_dict(model_state["model_state_dict"], strict=False)

    # Load the images
    print("Finding bad predictions...")
    for idx, img_path in enumerate(os.listdir(os.path.join(cwd, "data_yolo", "test", "images"))[:]):
        
        image_path = os.path.join(cwd, "data_yolo", "test", "images", img_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transforms.ToTensor()(image)

        label_path = os.path.join(cwd, "data_yolo", "test", "labels", img_path.replace(".jpg", ".txt"))
        targets = []
        with open(label_path, "r") as f:
            for line in f:
                values = line.strip().split()
                if len(values) != 5:
                    continue
                class_id = int(values[0])
                x_center, y_center, width, height = map(float, values[1:])
                targets.append([0, class_id, x_center, y_center, width, height])
        targets = torch.tensor(targets)

        # Do the forward pass
        yolov5.eval()
        with torch.no_grad():
            pred = yolov5(image.unsqueeze(0))

        # Do inference
        mAP = compute_map(
            pred=pred,
            true=targets,
            num_classes=len(CLASSES)
        )
        if mAP < 0.95:
            print(f"Bad prediction for image {img_path} at index {idx}, with mAP: {mAP:.4f}")
        
            boxes = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.5)[0]
            boxes, confidences, class_ids = process_pred(boxes)
            visualize_pred(
                image=image.permute(1, 2, 0).numpy(),
                boxes=boxes,
                classes_id=class_ids,
                classes=CLASSES
            )

            break