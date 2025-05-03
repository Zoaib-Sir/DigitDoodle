# Digit Doodle

## Overview

Digit Doodle is an interactive web application that challenges users to solve simple arithmetic problems by drawing two-digit numbers. Users draw the tens and ones digits on separate HTML canvas elements, and a PyTorch-powered convolutional neural network predicts the digits in real time. The app then evaluates the user’s answer and provides immediate feedback.

## Screenshots
![Screenshot 2025-05-03 203309](https://github.com/user-attachments/assets/95acbc8d-2a2f-4536-86bd-f5301f54f2a4)
![Screenshot 2025-05-03 203330](https://github.com/user-attachments/assets/d4702c69-1a98-4632-8efd-3b0fdd930873)
![Screenshot 2025-05-03 203432](https://github.com/user-attachments/assets/dac6f67b-68bc-4fb9-b8d9-c16024eedd25)


## Features

- interactive dual-canvas interface for tens and ones digits
- random generation of addition, subtraction, and multiplication problems
- enhanced MNIST classifier for accurate handwriting recognition
- dynamic feedback modals indicating correct or incorrect answers
- automatic problem refresh after each submission



## Deployment

Access the live demo here: [https://digitdoodle.onrender.com](https://digitdoodle.onrender.com)



## Machine Learning Pipeline

Digit Doodle’s digit recognizer is an enhanced convolutional neural network trained on the MNIST dataset with data augmentation and a custom preprocessing pipeline.

### Data Augmentation

- random rotations up to 10°
- affine translations up to 10%
- random resized crops (80–120% scale)

### Preprocessing Steps

1. decode base64 image to grayscale
2. threshold and invert (white digit on black background)
3. apply morphological opening to remove noise
4. extract the largest contour and crop the region
5. resize to 20×20 pixels, preserving aspect ratio
6. center in a 28×28 canvas
7. normalize using MNIST mean and standard deviation

### Model Architecture

- two convolutional blocks:
  - each block: Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm → ReLU → MaxPool → Dropout(0.4)
  - filters: 1→32→64
- classifier head:
  - Linear(64×7×7 → 512) → BatchNorm → ReLU → Dropout(0.4)
  - Linear(512 → 256) → BatchNorm → ReLU → Dropout(0.4)
  - Linear(256 → 10)
- output: log softmax over 10 classes

training uses the AdamW optimizer, cross-entropy loss, and a reduce-on-plateau LR scheduler. the best model (lowest validation loss) is saved as `best_model.pt`.

## Performance Metrics

After training for 50 epochs on the MNIST dataset, the model achieved the following results on the final batch and test set:

```
Train Epoch: 50 [0/60000 (0%)]    Loss: 0.013709
Train Epoch: 50 [12800/60000 (21%)]    Loss: 0.084054
Train Epoch: 50 [25600/60000 (43%)]    Loss: 0.013258
Train Epoch: 50 [38400/60000 (64%)]    Loss: 0.026022
Train Epoch: 50 [51200/60000 (85%)]    Loss: 0.008027

Test set: Average loss: 0.0000, Accuracy: 9964/10000 (99.64%)
```

### Training Loss Curve

![Training Loss Curve](path/to/training_loss_curve.png)

### Validation Accuracy Curve

![Validation Accuracy Curve](path/to/validation_accuracy_curve.png)

## Usage

1. load the web page; a random problem appears
2. draw the tens digit on the left canvas and the ones digit on the right
3. click Submit Answer to send your drawing for prediction
4. view feedback in the modal and try the next problem

## Future Improvements

- support for multi-digit numbers beyond two places
- user accounts and progress tracking
- persistent storage of scores and history
- deployment on a scalable cloud platform with GPU inference

## Dependencies

see `requirements.txt` for a full list of packages, including flask, torch, torchvision, opencv-python, pillow, and numpy

## Repository Structure

```
digit-doodle/
├── app.py                             # flask server and API endpoints
├── templates/
│   └── index.html                     # frontend interface and client-side logic
├── best_model.pt                      # trained PyTorch model weights
├── enhanced_mnist_classifier.ipynb    # notebook for training and preprocessing
├── requirements.txt                   # python dependencies
└── render.yaml                        # optional deployment configuration
```

## Installation

1. clone the repository

   ```
   git clone https://github.com/yourusername/digit-doodle.git
   cd digit-doodle
   ```

2. create and activate a virtual environment

   ```
   python -m venv venv
   source venv/bin/activate       # linux/mac
   venv\Scripts\activate        # windows
   ```

3. install dependencies

   ```
   pip install -r requirements.txt
   ```

4. start the flask server

   ```
   python app.py
   ```

5. open [http://localhost:10000](http://localhost:10000) in your browser

## License

This project is available under the MIT License.
