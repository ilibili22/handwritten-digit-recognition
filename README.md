# Handwritten Digit Recognition using CNN

A deep learning project that uses a Convolutional Neural Network (CNN) to recognize handwritten digits (0–9) from images using the MNIST dataset.

## Overview

This project implements a CNN model using TensorFlow and Keras to classify grayscale images of handwritten digits.

The model learns visual patterns and features from the images through convolutional and pooling layers, then predicts the corresponding digit.

## Dataset

- **Dataset:** MNIST
- **Training images:** 60,000
- **Test images:** 10,000
- **Image dimensions:** 28 × 28 pixels
- **Color channels:** 1 (grayscale)
- **Number of classes:** 10 (digits 0–9)

The dataset is loaded automatically using TensorFlow Keras.

## Model Architecture

The CNN model consists of:

- Convolutional layers (`Conv2D`)
- Max pooling layers (`MaxPooling2D`)
- Global average pooling (`GlobalAveragePooling2D`)
- Fully connected layers (`Dense`)
- Dropout layers for regularization
- Softmax output layer for digit classification

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn
- Pandas

## Model Performance

The model was trained for 10 epochs.

| Metric | Result |
|---|---:|
| Training Accuracy | 98.41% |
| Validation Accuracy | 98.70% |
| Test Accuracy | **98.92%** |
| Test Loss | 0.0338 |

## Installation

1. Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/handwritten-digit-recognition.git
```

2. Navigate to the project directory:

```bash
cd handwritten-digit-recognition
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the main Python file:

```bash
python main.py
```

The MNIST dataset will be downloaded automatically when the project is run for the first time.

## Future Improvements

- Visualize model predictions on test images.
- Analyze incorrectly classified images.
- Experiment with different CNN architectures.
- Save and load the trained model.

## Author

Developed as a personal deep learning project to explore Convolutional Neural Networks and image classification.