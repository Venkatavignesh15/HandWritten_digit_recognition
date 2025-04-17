🧠 MNIST Handwritten Digit Recognition using Deep Learning (CNN)
📌 Project Overview
This is a Deep Learning project that implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to accurately classify handwritten digits from the MNIST dataset. The model achieves over 99% test accuracy, demonstrating the power of deep neural networks in image classification tasks.

📊 Dataset
Dataset: MNIST Handwritten Digits

Image Size: 28x28 pixels, grayscale

Number of Classes: 10 (Digits 0–9)

Training Samples: 60,000

Test Samples: 10,000

🧠 Deep Learning Architecture
Conv2D (32 filters, 3x3) + ReLU + MaxPooling2D

Conv2D (64 filters, 3x3) + ReLU + MaxPooling2D

Conv2D (64 filters, 3x3) + ReLU

Flatten

Dense Layer (128 neurons) + ReLU

Dropout (0.5)

Output Layer (10 neurons) + Softmax

🚀 Model Performance
Training Accuracy: ~99.30%

Test Accuracy: ~99.28%

Loss Function: Categorical Crossentropy

Optimizer: Adam

🛠️ Technologies Used
Python

TensorFlow / Keras (Deep Learning Framework)

NumPy

Matplotlib

Jupyter Notebook

📁 Project Structure

📦 mnist-digit-recognition/
├── mnist_cnn.ipynb       # Deep Learning notebook
├── mnist_cnn.h5          # Trained model file
├── README.md             # Project documentation

Edit
pip install tensorflow matplotlib numpy
Run the notebook:

bash
Copy
Edit
jupyter notebook mnist_cnn.ipynb
