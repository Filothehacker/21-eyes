## Introduction

This is the computer vision project of Filippo Focaccia, Tommaso Vezzoli and Giulio Pirotta. The aim of this project was to detect blackjack cards in overlapping and challenging scenarios through some of the most relevant models of the YOLO family.

## Repository structure

This is how the repository is organised:

```
    21-eyes/
    ├── configurations/
    ├── data_classification/
    │   ├── train/
    │   ├── test/
    │   ├── development/
    │   ├── backgrounds/        # Background DTD images for dataset creation
    │   └── original_cards/     # Original card images
    ├── data_yolo/
    │   ├── train/
    │   ├── development/
    │   └── test/
    ├── models/                 # Model weights
    ├── src/
    │   ├── <model_name>/
    │   │   ├── train.py
    │   │   └── test.py
    │   └── ...
    └── utils_data/
```

This is how the repository is organised:
- configurations: this folder contains all the yaml and json files with hyperparameters, configurations and classes considered in all the implementations.

- data_classification: here we find the data used for the classification pre-training of YOLOv1 split into train, test and development. In each subfolders there are images and labels, where each label is the card rapresented in the associated image. We also find two folders containing backgrounds and original cards used to create the classification dataset from scratch.

- data_yolo: contains the training data for the detection task split into train, development and test. Each subfolder contains images and labels in classic YOLO txt format.

- models: this folder contains weights for all the used models.

- src: here we have stored folders for each model we explored. Each folder has their respective train and test file and addictive files depending on the use of the model.

- utils_data: this folder contains the scripts used to create and resize images for both the classification and detection dataset.

## Usage

To run the live detection, first clone the repository and install the requirements:

```
git clone
cd 21-eyes
pip install -r requirements.txt
```

Then, run the script:

```bash
python src/yolov5/live_detection.py
```

potentially replacing the model checkpoint with the one you want to use.  
Note that the weights are not included in the repository.