from matplotlib import pyplot as plt
import os
import yaml


def visualize_pred(image, boxes, classes_id=None, classes=None):
    
    plt.imshow(image)
    ax = plt.gca()

    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = map(int, box)
        w = x_max-x_min
        h = y_max-y_min

        if classes_id is not None:
            class_id = classes_id[i]

        # Create a rectangle patch and write the name of the class
        rect = plt.Rectangle((x_min, y_min), w, h, linewidth=1, edgecolor="r", facecolor="none")
        ax.add_patch(rect)

        if classes_id is not None and classes is not None:
            ax.text(x_min-w/2, y_min+h/4, classes[class_id], color="white", fontsize=8,
                    bbox=dict(facecolor="red", alpha=0.4)
            )

    # Save the image with bounding boxes
    output_path = os.path.join(os.getcwd(), "predictions.jpg")
    plt.savefig(output_path)
    plt.show()
    plt.close()


if __name__ == "__main__":
    cwd = os.getcwd()
    
    image_path = os.path.join(cwd, "data_yolo", "train", "images", "000090528_jpg.rf.d50e89610e5c97c61632c290692f3e75.jpg")
    image = plt.imread(image_path)

    label_path = os.path.join(cwd, "data_yolo", "train", "labels", "000090528_jpg.rf.d50e89610e5c97c61632c290692f3e75.txt")
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

    classes_path = os.path.join(cwd, "configurations", "classes.yaml")
    with open(classes_path, "r") as f:
        classes = yaml.safe_load(f)["classes"]

    visualize_pred(image, boxes, classes_id, classes)