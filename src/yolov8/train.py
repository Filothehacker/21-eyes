import os
from ultralytics import YOLO


cwd = os.getcwd()
data_yaml_path = os.path.join(cwd, "configurations", "yolov8.yaml")
weights_path = os.path.join(cwd, "models", "yolov8n.pt")

# Load the model
model = YOLO(weights_path)

# Train
model.train(
    data=data_yaml_path,
    epochs=30,
    imgsz=416,
    batch=16,
    name="yolov8"
)

# Evaluate
metrics = model.val()

# Save the model
model.export(format="onnx")