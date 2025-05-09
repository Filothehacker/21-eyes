import cv2
import numpy as np
import os
from inference import process_pred
from yolov5_ultralytics.models.yolo import Model
from yolov5_ultralytics.utils.general import non_max_suppression
import torch
import yaml


def preprocess_frame(frame, img_size=416, device='cpu'):

    # Resize the image
    img = cv2.resize(frame, (img_size, img_size))
    img = img[:, :, ::-1].transpose(2, 0, 1)

    # Convert to array and normalize
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float() / 255.0
    img = img.unsqueeze(0).to(device)

    return img


def rescale_box(box, original_shape, img_size=416):

    gain_w = original_shape[1] / img_size
    gain_h = original_shape[0] / img_size
    gain = np.array([gain_w, gain_h, gain_w, gain_h])
    box = box*gain
    return box


def draw_boxes(frame, boxes, class_ids, classes):

    for box, class_id in zip(boxes, class_ids):
        scaled_box = rescale_box(box, frame.shape[:2])
        x1, y1, x2, y2 = scaled_box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = classes[class_id]
        cv2.putText(frame, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


def main():

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")
    cwd = os.getcwd()

    # Load the card classes
    classes_path = os.path.join(cwd, "configurations", "classes.yaml")
    with open(classes_path, "r") as f:
        classes = yaml.safe_load(f)
    CLASSES = classes["classes"]

    # Load the model
    print("Instantiating the model...")
    model_config_path = os.path.join(cwd, "src", "yolov5", "yolov5_ultralytics", "models", "yolov5s.yaml")
    model = Model(model_config_path, ch=3, nc=len(CLASSES))

    model_path = os.path.join(cwd, "models", "yolov5_best.pth")
    model_state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(model_state["model_state_dict"], strict=False)
    model = model.to(DEVICE)
    model.eval()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    print("Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Preprocess the frame and pass the image through the model
        image = preprocess_frame(frame, device=DEVICE)
        with torch.no_grad():
            pred = model(image)
        boxes = non_max_suppression(pred)[0]
        boxes, _, class_ids = process_pred(boxes)

        # Draw the boxes on the frame
        if boxes is not None:
            draw_boxes(frame, boxes, class_ids, CLASSES)
        cv2.imshow("Live Card Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()