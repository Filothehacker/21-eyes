from ultralytics import YOLO

# Define paths
data_yaml_path = "data_v8/data.yaml"
weights_path = "yolov8n.pt"  # Pre-trained weights for YOLOv8
# Load the YOLOv8 model
model = YOLO(weights_path)

# Train the model
model.train(
    data=data_yaml_path,  # Path to the data.yaml file
    epochs=30,            # Number of epochs to train
    imgsz=640,            # Image size
    batch=16,             # Batch size
    name="yolov8_playing_cards"  # Name of the training run
)

# Evaluate the model on the validation set
metrics = model.val()

# Export the model to ONNX or other formats if needed
model.export(format="onnx")