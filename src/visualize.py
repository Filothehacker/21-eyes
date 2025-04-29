from matplotlib import pyplot as plt
import os
import yaml


def visualize_prediction(image, boxes, classes_id, classes):
    
    plt.imshow(image)
    ax = plt.gca()

    for i, box in enumerate(boxes):
        x = box[0]*image.shape[1]
        y = box[1]*image.shape[0]
        w = box[2]*image.shape[1]
        h = box[3]*image.shape[0]
        x_min = (x - w/2)
        y_min = (y - h/2)
        class_id = classes_id[i]

        # Create a rectangle patch and write the name of the class
        rect = plt.Rectangle((x_min, y_min), w, h, linewidth=1, edgecolor="r", facecolor="none")
        ax.add_patch(rect)

        ax.text(x_min-w/2, y_min+h/4, classes[class_id], color="white", fontsize=8,
                bbox=dict(facecolor="red", alpha=0.4)
        )

    plt.show()


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

    visualize_prediction(image, boxes, classes_id, classes)