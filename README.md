# MNIST Digit Recognition + Drawing GUI
This project implements a Convolutional Neural Network (CNN) trained on the MNIST dataset to recognize handwritten digits. \
Instead of providing an image file, the application opens a drawing window where the user can draw a digit. The model then processes the drawing in real time and predicts the digit.

## Features
- CNN trained on MNIST dataset
- ~99% validation accuracy
- Live drawing canvas using OpenCV
- Real-time digit prediction
- MNIST-style preprocessing pipeline
- Automatic centering and normalization
- Best model checkpoint saving

## How It Works
- User draws a digit on the canvas.
- The drawing is converted to grayscale.
- Image is thresholded and cleaned.
- Digit is resized to 28x28 pixels.
- Center of mass alignment is applied.
- Image is normalized using MNIST statistics.
- CNN predicts the digit.

## Requirements
- Python 3.9+
- PyTorch
- Torchvision
- OpenCV
- NumPy
