# Coin Image Analysis

This repository contains code for analyzing images of coins to detect and classify them. It includes various image processing techniques and models for coin detection and classification.

## Dataset

The image dataset used for training and testing the models can be found in the following Google Drive link: [Coin Image Dataset](https://drive.google.com/drive/folders/1gaUQiRAkI55kTb-vYtrsZdYuzPLKVYel)

## Image Preprocessing

The image preprocessing steps are as follows:
1. Reading the images dataset.
2. Remove Glare from the images.
3. Gaussian blurring to reduce noise.
4. Adaptive histogram equalization to enhance contrast.
5. Binarization to separate the coins from the background.
6. Morphological operations to remove noise and fill gaps.
7. Contour detection to find the coins in the image using the Canny edge detector.
8. Ellipse fitting to the contours to approximate the coins' shapes.
9. Filtering ellipses based on size, shape, and intersection over union.
10. Remove overlapping ellipses by keeping the largest one.
11. Extracting regions of interest (ROIs) from the ellipses.

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

## Model Training

The model training process is implemented in a Jupyter Notebook named `cnn_coin_classifier.ipynb`. This notebook contains code for loading the dataset, preprocessing the images, training the model, and evaluating its performance.
Using a CNN model with the following architecture:
Architecture:
- Convolutional layer with 32 filters, kernel size (3, 3), and ReLU activation.
- Max pooling layer with pool size (2, 2).
- Convolutional layer with 64 filters, kernel size (3, 3), and ReLU activation.
- Max pooling layer with pool size (2, 2).
- Convolutional layer with 128 filters, kernel size (3, 3), and ReLU activation.
- Max pooling layer with pool size (2, 2).
- Flatten layer.
- Dense layer with 128 units and ReLU activation.
- Dropout layer with a dropout rate of 0.5.
- Dense layer with the number of classes and softmax activation.
Since the dataset is small, we use data augmentation to generate more training samples. The model is trained using the Adam optimizer and categorical crossentropy loss function.
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
