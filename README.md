# Coin Image Analysis

This repository contains code for analyzing images of coins to detect and classify them. It includes various image processing techniques and models for coin detection and classification.

## Dataset

The image dataset used for training and testing the models can be found in the following Google Drive link: [Coin Image Dataset](https://drive.google.com/drive/folders/1gaUQiRAkI55kTb-vYtrsZdYuzPLKVYel)

## Image Preprocessing

The image preprocessing steps are as follows:
1. Reading the images dataset.
2. Removing glare from the images.
3. Applying Gaussian blurring to reduce noise.
4. Performing adaptive histogram equalization to enhance contrast.
5. Binarizing the images to separate the coins from the background.
6. Utilizing morphological operations to remove noise and fill gaps.
7. Detecting contours to identify the coins in the image using the Canny edge detector.
8. Fitting ellipses to the contours to approximate the shapes of the coins.
9. Filtering ellipses based on their size, shape, and intersection over union.
10.Removing overlapping ellipses by retaining the largest one.
11. Extracting regions of interest (ROIs) from the ellipses.


## Dataset Organization: 

1. Data Cleaning: Unnecessary or missing data in the dataset has been removed.
2. Data Transformation: Data types or formats have been transformed. Euro and cent values have been concatenated to form comma-separated numbers. 
3. Euro and centime units have been removed from the values.
4. Data Splitting: The dataset has been split into training, validation, and test subsets. This allows for evaluating the model's training using the reserved test data and adjusting the model's hyperparameters using the validation data.

## Coin Detection

### Method 1: Feature Matching 
Feature matching entails the following steps:

1. **Constructing a Vocabulary using SIFT Descriptors:** The first step involves identifying unique features in both images and creating a list of "words" using SIFT descriptors. SIFT descriptors are vectors that describe features.

2. **Matching Descriptors using the FLANN Matcher:** The generated SIFT descriptors are matched with similar features in the second image. FLANN is a fast library for finding approximate nearest neighbors.

3. **Filtering Matches using Lowe's Ratio Test:** At this stage, Lowe's ratio test is applied to select reliable matches among all matches. This test evaluates how well the best match agrees with the second-best match, filtering out unreliable matches.

4. **Estimating Homography using RANSAC:** Finally, RANSAC (Random Sample Consensus) is employed to estimate the homography between the matched feature points. RANSAC is a robust algorithm used to estimate parameters of a mathematical model from a set of observed data points.

### Method 2: CNN Classification 
Coin detection involves the following steps:

1. **Loading the CNN Model:** The pre-trained Convolutional Neural Network (CNN) model for coin detection is loaded.

2. **Classifying Regions of Interest (ROIs) using CNN:** Regions of interest (ROIs), identified from the images, are passed through the CNN model to classify and detect coins accurately.


### Method 1: Feature Matching
**Feature matching involves:**
- Build vocabulary using SIFT descriptors.
- Match descriptors using FLANN matcher.
- Filter matches using Lowe's ratio test.
- Estimate homography using RANSAC.

### Method 2: CNN Classification
**Coin detection involves:**
- Load the CNN model.
- Classifying ROIs using CNN for coin detection.


## Model Training 

The model training process is implemented in a Jupyter Notebook named `cnn_coin_classifier.ipynb`. This notebook contains code for loading the dataset, preprocessing the images, training the model, and evaluating its performance.

The CNN model architecture used for coin classification is as follows:

1. Convolutional layer with 32 filters, kernel size (3, 3), and ReLU activation.
2. Max pooling layer with pool size (2, 2).
3. Convolutional layer with 64 filters, kernel size (3, 3), and ReLU activation.
4. Max pooling layer with pool size (2, 2).
5. Convolutional layer with 128 filters, kernel size (3, 3), and ReLU activation.
6. Max pooling layer with pool size (2, 2).
7. Flatten layer.
8. Dense layer with 128 units and ReLU activation.
9. Dropout layer with a dropout rate of 0.5.
10. Dense layer with the number of classes and softmax activation.

Since the dataset is small, data augmentation techniques are applied to generate more training samples. The model is trained using the Adam optimizer and categorical crossentropy loss function.

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
The evaluation process includes several steps aimed at assessing the performance of both CNN and feature matching techniques:

**1. Prediction on Test Images:**
- Use the trained CNN model to predict the coin classes for the test images.
 - Applying feature matching techniques to detect coins in the test images.

**2. Calculation of Accuracy for Detected Coins:**
 - Compare the estimated coin classes obtained from the CNN model with ground truth labels to calculate accuracy.
 - Evaluating the accuracy of coins detected using feature matching against manually annotated ground truth information.

**3. Mean Absolute Error (MAE) Calculation for Total Coin Value:**
 - Calculate the total value of coins detected using both CNN and feature matching techniques.
 - Calculate the absolute difference between the predicted total value and the ground truth total value for each image
 - Averaging these absolute differences across all test images to obtain the MAE.

**4. Performance Metrics Analysis:**
 - Analyse the accuracy and MAE results to evaluate the effectiveness of both techniques.
 - Identify the strengths or weaknesses of each method based on performance metrics


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
