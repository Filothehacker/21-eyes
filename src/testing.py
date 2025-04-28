# # import the .pth file to upload the trained model
# from resnet import YOLOv1ResNet
# from dotenv import load_dotenv
# import json
# import os
# import torch
# from inference import *
# from utils_yolo import eval
# from visualize import visualize_prediction
# import yaml
# from PIL import Image
# from torchvision import transforms
# import matplotlib.pyplot as plt



# if __name__ == "__main__":

#     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#     cwd = os.getcwd()
#     model_config_path = os.path.join(cwd, "configurations", "yolo_v1.json")
#     with open(model_config_path, "r") as f:
#         model_config = json.load(f)

#     # Retrieve the parameters
#     MODEL_PARAMS = model_config["MODEL_PARAMS"]

#     model = YOLOv1ResNet(model_params=MODEL_PARAMS)
#     model_path = os.path.join(cwd, "models", "yolo_v1.pth")
#     print("Loading the model from", model_path)
    
#     # Load the complete checkpoint dictionary
#     checkpoint = torch.load(model_path)
    
#     # Extract just the model state dictionary
#     model.load_state_dict(checkpoint["model_state_dict"])
    
#     print("Model loaded successfully.")
#     model.to(DEVICE)
#     model.eval()


#     classes_path = os.path.join(cwd, "configurations", "classes.yaml")
#     with open(classes_path, "r") as f:
#         classes = yaml.safe_load(f)["classes"]
    
#     image = Image.open(os.path.join(cwd, "data", "development", "images", "000507247_jpg.rf.6a57e870859691da0fcd008760344bcc.jpg"))
#     image = image.convert("RGB")
#     original_image = plt.imread(os.path.join(cwd, "data", "development", "images", "000507247_jpg.rf.6a57e870859691da0fcd008760344bcc.jpg"))
#     # Apply the same transforms as in the dataset
#     # Use development dataset mean and std values
#     mean_ = [0.55623617, 0.49624988, 0.45092961]
#     std_ = [0.20111711, 0.20110672, 0.2046315]
    
#     transform = transforms.Compose([
#         transforms.Resize((448, 448)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=mean_, std=std_)
#     ])

#     image = transform(image).unsqueeze(0).to(DEVICE)


#     #Run inference
#     with torch.no_grad():
#         predictions = model(image)
    
#     S = MODEL_PARAMS["S"]
#     B = MODEL_PARAMS["B"]
#     confidence_threshold = 0.5
#     iou_threshold = 0.5
#     processed_preds = process_pred(predictions, B)

#     print("Processed predictions shape:", processed_preds.shape)
#     # remove the batch dimension
#     processed_preds = processed_preds.squeeze(0)
#     print("Processed predictions shape after squeeze:", processed_preds.shape)

#     pred_boxes, pred_confidences, pred_classes = convert_boxes_to_list(processed_preds, S, B, resize=True)
#     nms_boxes, nms_confidences, nms_classes = apply_non_max_suppression(pred_boxes, pred_confidences, pred_classes, confidence_threshold, iou_threshold)

#     # Visualize results
#     print(f"Found {len(nms_boxes)} objects with confidence > {confidence_threshold}")
#     for i, (box, conf, cls) in enumerate(zip(nms_boxes, nms_confidences, nms_classes)):
#         print(f"Object {i+1}: Class = {classes[cls]} ({cls}), Confidence = {conf:.4f}")
#         print(f"  Box = (x={box[0]:.4f}, y={box[1]:.4f}, w={box[2]:.4f}, h={box[3]:.4f})")
    
#     visualize_prediction(original_image, nms_boxes, nms_classes, classes)
#     plt.show()