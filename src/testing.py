# import the .pth file to upload the trained model
from resnet import YOLOv1ResNet
from dotenv import load_dotenv
import json
import os
import torch
from inference import *
from utils_yolo import eval
from visualize import visualize_prediction
import yaml
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from random import choice
 
 
 
if __name__ == "__main__":
 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
 
    cwd = os.getcwd()
    model_config_path = os.path.join(cwd, "configurations", "yolo_v1.json")
    with open(model_config_path, "r") as f:
        model_config = json.load(f)
 
    # Retrieve the parameters
    MODEL_PARAMS = model_config["MODEL_PARAMS"]
 
    model = YOLOv1ResNet(model_params=MODEL_PARAMS)
    model_path = os.path.join(cwd, "models", "yolo_v1.pth")
    print("Loading the model from", model_path)
   
    # Load the complete checkpoint dictionary
    checkpoint = torch.load(model_path,map_location=DEVICE)
   
    # Extract just the model state dictionary
    model.load_state_dict(checkpoint["model_state_dict"])
   
    print("Model loaded successfully.")
    model.to(DEVICE)
    model.eval()
 
 
    classes_path = os.path.join(cwd, "configurations", "classes.yaml")
    with open(classes_path, "r") as f:
        classes = yaml.safe_load(f)["classes"]
   
    #pick a random path from the dataset
    img_path = os.path.join(cwd, "data", "development", "images")
    random_image = choice(os.listdir(img_path))
    image_path = os.path.join(img_path, random_image)
 
 
    # image = Image.open(os.path.join(cwd, "data", "train", "images", "010010692_jpg.rf.4490c8593eefaab0bfd1bfeca898e594.jpg"))
    # image = image.convert("RGB")
    # original_image = plt.imread(os.path.join(cwd, "data", "train", "images", "010010692_jpg.rf.4490c8593eefaab0bfd1bfeca898e594.jpg"))
    image = Image.open(image_path)
    image = image.convert("RGB")
    original_image = plt.imread(image_path)
    original_image = transforms.Resize((448, 448))(Image.fromarray((original_image).astype('uint8')))
    original_image = np.array(original_image)
 
 
    label_path = image_path.replace("images", "labels").replace(".jpg", ".txt")
    boxes = []
    classes_id = []
    with open(label_path, "r") as f:
            for line in f:
                values = line.strip().split()
                if len(values) != 5:
                    continue
                class_id = int(values[0])
                x_center, y_center, width, height = map(float, values[1:])
                boxes.append([x_center, y_center, width, height])
                classes_id.append(class_id)
    visualize_prediction(original_image, boxes, classes_id, classes)
    output_dir = os.path.join(cwd, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"original_test.png")
    plt.savefig(output_path)
 
    # Apply the same transforms as in the dataset
    # Use development dataset mean and std values
    # mean_ = [0.55623617, 0.49624988, 0.45092961]
    # std_ = [0.20111711, 0.20110672, 0.2046315]
    mean_ = [0.55439766, 0.49272217, 0.44703565]
    std_ = [0.20015266, 0.20010346, 0.20344909]
   
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_, std=std_)
    ])
 
    image = transform(image).unsqueeze(0).to(DEVICE)
 
 
    #Run inference
    with torch.no_grad():
        predictions = model(image)
   
    S = MODEL_PARAMS["S"]
    B = MODEL_PARAMS["B"]
    confidence_threshold = 0.8
    iou_threshold = 0.8
    print(predictions.shape)
    processed_preds = process_pred(predictions, B)
 
    print("Processed predictions shape:", processed_preds.shape)
    # remove the batch dimension
    processed_preds = processed_preds.squeeze(0)
    print("Processed predictions shape after squeeze:", processed_preds.shape)
 
    pred_boxes, pred_confidences, pred_classes = convert_boxes_to_list(processed_preds, S, B, resize=True)
    nms_boxes, nms_confidences, nms_classes = apply_non_max_suppression(pred_boxes, pred_confidences, pred_classes, confidence_threshold, iou_threshold)
    # print(len(nms_boxes), len(nms_confidences), len(nms_classes))
    # Visualize results
    # print(f"Found {len(nms_boxes)} objects with confidence > {confidence_threshold}")
    # for i, (box, conf, cls) in enumerate(zip(nms_boxes, nms_confidences, nms_classes)):
        # print(cls)
        # print(f"Object {i+1}: Class = {classes[cls]} ({cls}), Confidence = {conf:.4f}")
        # print(f"  Box = (x={box[0]:.4f}, y={box[1]:.4f}, w={box[2]:.4f}, h={box[3]:.4f})")
   
    visualize_prediction(original_image, nms_boxes, nms_classes, classes)
    output_dir = os.path.join(cwd, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"detection_test.png")
    plt.savefig(output_path)