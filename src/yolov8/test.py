import cv2
import os
import yaml
from ultralytics import YOLO
import matplotlib.pyplot as plt
from torchvision import transforms


def visualize_pred(image, boxes, classes_id=None, classes=None, confidences=None):
    
    plt.imshow(image)
    ax = plt.gca()
    plt.axis("off")

    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = map(int, box)
        w = x_max-x_min
        h = y_max-y_min

        label = ""
        if classes_id is not None:
            class_id = classes_id[i]
            label += f"{classes[class_id]}"
            if confidences is not None:
                label += f" {confidences[i]:.2f}"
        else:
            label += f"?"

        # Create a rectangle patch and write the name of the class
        rect = plt.Rectangle((x_min, y_min), w, h, linewidth=1, edgecolor="r", facecolor="none")
        ax.add_patch(rect)

        if classes_id is not None and classes is not None:
            ax.text(x_min-w/2, y_min+h/4, label, color="white", fontsize=8,
                    bbox=dict(facecolor="red", alpha=0.4)
            )

    # Save the image with bounding boxes
    output_path = os.path.join(os.getcwd(), "predictions.jpg")
    plt.savefig(output_path)
    plt.show()


if __name__ == "__main__":
    cwd = os.getcwd()

    # Load the classes
    classes_path = os.path.join(cwd, "configurations", "classes.yaml")
    with open(classes_path, "r") as f:
        classes = yaml.safe_load(f)
    CLASSES = classes["classes"]

    # Load the image
    print("Loading an image...")
    image_path = os.path.join(cwd, "data_yolo", "test", "images", "0.jpg")
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (416, 416))
    image = transforms.ToTensor()(image)

    # Load the model
    print("Instantiating the model...")
    model_path = os.path.join(cwd, "models", "yolov8_best.pt")
    yolov8 = YOLO(model_path)

    # Do the forward pass
    print("Passing the image through the model...")
    results = yolov8(image_path, verbose=False)

    print("Doing inference...")
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes_id = results[0].boxes.cls.cpu().numpy().astype(int)
    confidences = results[0].boxes.conf.cpu().numpy()

    visualize_pred(
        image=image.permute(1, 2, 0).numpy(),
        boxes=boxes,
        classes_id=classes_id,
        classes=CLASSES,
        confidences=confidences
    )