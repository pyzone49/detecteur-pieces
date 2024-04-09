# Coin Image Analysis

This repository contains code for analyzing images of coins to detect and classify them. It includes various image processing techniques and models for coin detection and classification.

## Dataset

The image dataset used for training and testing the models can be found in the following Google Drive link: [Coin Image Dataset](https://drive.google.com/drive/folders/1gaUQiRAkI55kTb-vYtrsZdYuzPLKVYel)

## Model Training

The model training process is implemented in a Jupyter Notebook named `cnn_coin_classifier.ipynb`. This notebook contains code for loading the dataset, preprocessing the images, training the model, and evaluating its performance.
Using a CNN model with the following architecture:
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
Since the dataset is small, we use data augmentation to generate more training samples. The model is trained using the Adam optimizer and categorical crossentropy loss function.

## Image Preprocessing

The image preprocessing steps are as follows:
1. Reading the images dataset.
2. Applying various preprocessing techniques such as converting to grayscale, Gaussian blur, adaptive histogram equalization, binarization, morphological opening and closing, and edge detection using Canny.

## Feature Extraction

Feature extraction techniques used include:
- Harris-Laplace keypoints detection.
- Ellipse fitting to contours.
- Filtering ellipses based on size, shape, and intersection over union.
- Extracting regions of interest (ROIs) from ellipses.
- Extracting features from ROIs using Convolutional Neural Networks (CNN).

## Coin Detection

### Method 1: Feature Matching
Feature matching involves:
- Build vocabulary using SIFT descriptors.
- Match descriptors using FLANN matcher.
- Filter matches using Lowe's ratio test.
- Estimate homography using RANSAC.

### Method 2: CNN Classification
Coin detection involves:
- Load the CNN model.
- Classifying ROIs using CNN for coin detection.

## Evaluation
The evaluation process includes:
- Predicting on test images using both CNN and feature matching techniques.
- Calculating accuracy for the number of detected coins.
- Computing the mean absolute error for the total value of the detected coins.
- Analyzing the results and generating performance metrics.

## Requirements

To run the code in this repository, you need the following dependencies:
- Python 3.x
- OpenCV
- NumPy
- Matplotlib
- TensorFlow
- Pandas
- Jupyter Notebook

You can install the required packages using pip:
```bash
pip install -r requirements.txt
```
