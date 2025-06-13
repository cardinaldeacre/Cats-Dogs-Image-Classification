# Cats & Dogs Image Classifier

A deep learning project to classify images as either 'cat' or 'dog' using a pre-trained ResNet-18 model and PyTorch. Demonstrates efficient image classification through transfer learning.

## 🚀 Project Overview

The main goal of this project is to build an automated system that can accurately classify unseen images into one of two categories: "Cat" or "Dog". We leverage the power of **Transfer Learning** with a pre-trained **ResNet-18** model in PyTorch, allowing for high accuracy with efficient training.

## ✨ Features

-   **Deep Learning Classification:** Uses a Convolutional Neural Network (CNN) for image recognition.
-   **Transfer Learning:** Employs a pre-trained ResNet-18 model (trained on ImageNet) to accelerate learning and improve performance, especially with limited dataset size.
-   **Custom Dataset Handling:** Includes a PyTorch `Dataset` class to efficiently load and preprocess image data.
-   **Training & Prediction:** Provides functions for training the model and making predictions on individual images.
-   **Visual Prediction:** Displays the input image along with its predicted label.

## 🛠️ Technologies Used

* **Python**
* **PyTorch** (for deep learning framework)
* **torchvision** (for ResNet-18 model and image transforms)
* **Pillow (PIL)** (for image handling)
* **Matplotlib** (for image visualization)

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/cardinaldeacre/Cats-Dogs-Image-Classification.git    
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Linux/macOS:
    source venv/bin/activate
    # On Windows:
    .\venv\Scripts\activate
    ```
3.  **Install the required libraries:**
    ```bash
    pip install torch torchvision matplotlib pillow
    ```
    *(**Note:** Ensure you install the correct PyTorch version for your system, especially if using CUDA/GPU. Refer to the official [PyTorch website](https://pytorch.org/get-started/locally/) for detailed instructions.)*

## 🚀 Usage

1.  **Place your dataset:**
    Ensure your `cat-and-dog` dataset is structured as described in the `Dataset Structure` section. If running on platforms like Kaggle, adjust the `root_dir` path in the `KucingAnjingDataset` class accordingly (e.g., `/kaggle/input/cat-and-dog/training_set/training_set`).

2.  **Run the main script:**
    The provided Python code (which can be saved as, for example, `main.py`) will first train the model and then perform a sample prediction.

    ```bash
    python main.py
    ```

    -   The `train_model()` function will initiate the training process using the specified dataset, saving the trained model's weights to `cat_dog_checkpoint.pth`.
    -   After training, the `predict_image()` function will load this saved model and demonstrate a classification on a sample image from the test set, displaying the image with its predicted label.

## 📊 Training Results

During training, the model's **loss** (error) was consistently monitored. We observed a **clear decrease in the loss value across epochs**, indicating that the model successfully learned and improved its ability to distinguish between cat and dog images. This demonstrates the effective adaptation of the pre-trained knowledge to our specific binary classification task.

## 🙏 Acknowledgements

* This project utilizes the "cat-and-dog" dataset, commonly available on platforms like Kaggle.
* Built with the powerful [PyTorch](https://pytorch.org/) deep learning framework.
