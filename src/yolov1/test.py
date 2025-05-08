import cv2
from inference import process_pred, convert_boxes_to_list, apply_non_max_suppression
import json
import os
from torchvision import transforms
import torch
import yaml
from yolov1 import YoloV1
from visualize import visualize_pred


if __name__ == '__main__':
    cwd = os.getcwd()

    # Load the model configuration file
    print("Reading the configuration files...")
    model_config_path = os.path.join(cwd, "configurations", "yolov1.json")
    with open(model_config_path, "r") as f:
        model_config = json.load(f)

    # Retrieve the parameters
    MODEL_PARAMS = model_config["MODEL_PARAMS"]
    CNN_DICT = model_config["CNN"]
    MLP_DICT = model_config["MLP"]
    OUTPUT_SIZE = MODEL_PARAMS["S"]*MODEL_PARAMS["S"] * (MODEL_PARAMS["B"]*5+MODEL_PARAMS["C"])
    MLP_DICT["out_size"] = OUTPUT_SIZE

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

    # Instantiate the model
    print("Instantiating the model...")
    yolo_v1 = YoloV1(
        model_params=MODEL_PARAMS,
        cnn_blocks=CNN_DICT,
        mlp_dict=MLP_DICT
    )

    # Load the model weights
    model_path = os.path.join(cwd, "models", "yolov1_best.pth")
    model_state = torch.load(model_path)
    yolo_v1.load_state_dict(model_state["model_state_dict"], strict=False)

    # Do the forward pass
    print("Passing the image through the model...")
    yolo_v1.eval()
    with torch.no_grad():
        output = yolo_v1(image.unsqueeze(0))

    # Do inference
    print("Doing inference...")
    pred_processed = process_pred(output, MODEL_PARAMS['B'])
    pred_boxes, pred_confidences, pred_classes = convert_boxes_to_list(pred_processed[0], MODEL_PARAMS['S'])

    new_boxes, new_confidences, new_classes = apply_non_max_suppression(
        boxes=pred_boxes,
        confidences=pred_confidences,
        classes=pred_classes,
        confidence_threshold=0.05,
        iou_threshold=0.05
    )
    
    visualize_pred(
        image=image.permute(1, 2, 0).numpy(),
        boxes=new_boxes,
        classes_id=new_classes,
        classes=CLASSES
    )
