\# 🔢 Handwritten Digit Recognition using CNN



A beginner-friendly Artificial Intelligence project that recognizes handwritten digits from \*\*0 to 9\*\* using a Convolutional Neural Network (CNN).



\## 📌 Project Overview



This project uses the \*\*MNIST handwritten digit dataset\*\* to train a CNN model.



The trained model can recognize handwritten digits drawn by the user through a Streamlit web application.



\## 🎯 Objective



The main objectives of this project are:



\- Understand image classification.

\- Learn how CNNs work with image data.

\- Train a neural network using the MNIST dataset.

\- Build a simple real-time prediction application.

\- Display the predicted digit and confidence score.



\## 🧠 Technologies Used



\- Python

\- TensorFlow

\- Keras

\- NumPy

\- Pillow

\- Streamlit

\- Streamlit Drawable Canvas

\- CNN (Convolutional Neural Network)



\## 📊 Dataset



The project uses the \*\*MNIST handwritten digit dataset\*\*.



The dataset contains:



\- 60,000 training images

\- 10,000 testing images

\- 10 digit classes: 0–9

\- Image size: 28 × 28 pixels



\## 🏗️ CNN Architecture



The model contains:



1\. Input Layer

2\. Convolutional Layer — 32 filters

3\. Max Pooling Layer

4\. Convolutional Layer — 64 filters

5\. Max Pooling Layer

6\. Flatten Layer

7\. Dense Layer — 128 neurons

8\. Dropout Layer

9\. Output Layer — 10 classes



The output layer uses \*\*Softmax\*\* to calculate the probability of each digit.



\## 🔄 Project Workflow



```text

MNIST Dataset

&#x20;     ↓

Image Preprocessing

&#x20;     ↓

Pixel Normalization

&#x20;     ↓

CNN Model

&#x20;     ↓

Model Training

&#x20;     ↓

Model Evaluation

&#x20;     ↓

Saved Model

&#x20;     ↓

Streamlit Application

&#x20;     ↓

User Draws Digit

&#x20;     ↓

Image Preprocessing

&#x20;     ↓

CNN Prediction

&#x20;     ↓

Predicted Digit + Confidence

