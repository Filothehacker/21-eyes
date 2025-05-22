### Introduction
This is the computer vision project of Filippo Focaccia, Tommaso Vezzoli and Giulio Pirotta. The aim of this project was to detect blackjack cards in overlapping and challenging scenarios through some of the most relevant models of the YOLO family.

### Folders
This is how the repository is organised:
- configurations: this folder contains all the yaml, json files which contain hyperparameters, configurations and classes considered in all the implementations.

- data_classification: here we find the data which has been used for the classification pretatraing for the YOLOv1 model split into train, test and development.In each of these folders we find images and labels, where each label is the card rapresented in the associated image. We also find two folders containing backgrounds and original cards used to create the classification dataset from scratch.

- data_yolo: contains the training data for the detection task split into train, development and test. Each one of these folders contains images and labels in classic YOLO txt format.

- models: this folder contains weights for all the used models.

- notebooks: here we have the notebook used to create the card detection dataset.

- src: here we have stored folders for each model we explored. Each folder has their respective train and test file and addictive files depending on the use of the model.

- utils_data: this folder contains two python scripts used to create and resize images for the classification dataset.
