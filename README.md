# Melanoma Detection Assignment

This project focuses on skin cancer classification using CNNs (Convolutional Neural Networks). The dataset used is from the Skin Cancer ISIC The International Skin Imaging Collaboration. Data augmentation and techniques such as dropout, batch normalization, and L2 regularization have been applied to improve model performance and reduce overfitting.

## Table of Contents
- [General Information](#general-information)
- [Technologies Used](#technologies-used)
- [Data Augmentation](#data-augmentation)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## General Information
The project involves building a CNN model to classify images of skin lesions into different types of skin cancer. The dataset contains various classes of skin lesions, and the goal is to build a robust model that can classify unseen data accurately.

- **Objective**: To build a CNN-based classification model for skin lesion images.
- **Dataset**: The images were sourced from the ISIC dataset.
- **Challenge**: The dataset was imbalanced, and thus data augmentation techniques were applied to generate synthetic samples.

## Technologies Used
- **TensorFlow** - version 2.17.0
- **Augmentor** - version 0.2.12
- **NumPy** - version 1.26.4
- **Matplotlib** - version 3.7.1
- **Pandas** - version 2.1.4

## Data Augmentation
To handle class imbalance and improve model generalization, the following data augmentation techniques were applied:

- Random Horizontal and Vertical Flips
- Random Rotation (up to 20 degrees)
- Random Zoom (20%)
- Random Contrast Adjustment

## Model Architecture
The CNN model consists of several convolutional blocks with batch normalization and dropout layers to prevent overfitting. The model uses four convolutional layers, followed by fully connected dense layers, and a softmax output layer for multi-class classification.

### Model Summary:
- **Input**: 180x180x3 images (rescaled between 0-1)
- **4 Convolutional Layers**: 32, 64, 128, and 256 filters
- **MaxPooling2D layers** after each convolution
- **Dropout** (30-50%) and L2 regularization to prevent overfitting
- **Dense Layer** with 256 units and Dropout
- **Output Layer**: Softmax activation for multi-class classification

### Loss Function:
- `Sparse Categorical Crossentropy` was used as the loss function because the labels are integers.

### Optimizer:
- `Adam` optimizer with a learning rate schedule.

## Acknowledgements
- The project was inspired by the ISIC skin cancer classification challenge.

## Contact
Created by [@PranavSingh](https://github.com/psingh-2503) - feel free to reach out if you have any questions!
