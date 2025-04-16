from matplotlib import pyplot as plt
import os
import yaml


def visualize_prediction(image, boxes, classes_id, classes):
    
    plt.imshow(image)
    ax = plt.gca()

    for i, box in enumerate(boxes):
        x, y, w, h = box
        x_min = (x - w/2) * image.shape[1]
        y_min = (y - h/2) * image.shape[0]
        class_id = classes_id[i]

        # Create a rectangle patch and write the name of the class
        rect = plt.Rectangle((x_min, y_min), w*image.shape[1], h*image.shape[0], linewidth=2, edgecolor="r", facecolor="none")
        ax.add_patch(rect)

        ax.text(x_min, y_min, classes[class_id], color="white", fontsize=12,
                bbox=dict(facecolor="red", alpha=0.5))

    plt.show()


if __name__ == "__main__":
    
    cwd = os.getcwd()
    img_path = os.path.join(cwd, "data/train/images/000090528_jpg.rf.d50e89610e5c97c61632c290692f3e75.jpg")
    label_path = os.path.join(cwd, "data/train/labels/000090528_jpg.rf.d50e89610e5c97c61632c290692f3e75.txt")
    classes_path = os.path.join(cwd, "configurations/classes.yaml")

    img = plt.imread(img_path)
    boxes = [
        [0.4603365384615384, 0.6935096153846154, 0.06490384615384616, 0.0420673076923077],
        [0.6442307692307693, 0.5012019230769231, 0.06370192307692307, 0.0889423076923077],
        [0.6141826923076923, 0.07451923076923077, 0.06490384615384616, 0.0889423076923077]
    ]
    classes_id = [51, 37, 37]
    with open(classes_path, "r") as f:
        classes = yaml.safe_load(f)["classes"]

    visualize_prediction(img, boxes, classes_id, classes)