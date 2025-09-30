# Streamlit App for Handwritten Digit Recognition

## Overview
This repository contains a Streamlit web application for handwritten digit recognition.  
The app uses a trained CNN model (from the MNIST dataset) to predict digits from uploaded images or a drawing canvas.


## Features
- Upload **single or multiple digit images** for prediction.
- Draw digits on an interactive canvas.
- Displays predicted digit with confidence score.
- Shows probability distribution of all classes as a bar chart.
- Professional and user-friendly UI.



## Project Workflow
1. Load pre-trained CNN model (`mnist_cnn.h5`).
2. Preprocess inputs:
   - Convert to grayscale.
   - Resize to 28x28.
   - Normalize pixel values.
   - Auto-invert if background/foreground mismatch.
3. Predict using CNN.
4. Display results and probabilities in Streamlit UI.



## Technologies Used
- Python
- Streamlit
- TensorFlow / Keras
- NumPy
- OpenCV
- Pillow
- streamlit-drawable-canvas


## How to Run
### 1. Clone the repository
```bash
git clone https://github.com/your-username/mnist-digit-recognition-app.git
cd mnist-digit-recognition-app

## Deployment

This app can be deployed easily on:
Hugging Face Spaces

Author
Anam Jafar

