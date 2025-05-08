import cv2
from inference import process_pred
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