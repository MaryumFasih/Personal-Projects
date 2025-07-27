# Multimodal Housing Price Prediction

This project implements a multimodal deep learning model to predict house prices using both tabular data and images. The model combines a convolutional neural network (CNN) for image processing with a multilayer perceptron (MLP) for tabular data, leveraging PyTorch and scikit-learn.

## Table of Contents

* Overview
* Features
* Requirements
* Setup
* Data Preparation
* Model Architecture
* Training
* Evaluation
* Visualization
* Usage
* File Structure
* Contributing

## Overview

The `multimodal_housing.py` script trains a neural network to predict house prices by combining image data (house photos) with tabular features (e.g., lot area, overall quality). It uses a pre-trained ResNet18 model for image feature extraction and an MLP for tabular data, merging both for final predictions.

## Features

* Multimodal Learning: Combines image and tabular data for robust predictions.
* Pre-trained CNN: Utilizes ResNet18 with ImageNet weights for image feature extraction.
* Data Preprocessing: Scales tabular features and applies image transformations (resize, tensor conversion).
* Training and Evaluation: Implements training with Adam optimizer and evaluates using MAE and RMSE metrics.
* Visualization: Plots true vs. predicted house prices for model performance analysis.

## Requirements

* Python 3.8+
* PyTorch
* torchvision
* pandas
* numpy
* scikit-learn
* matplotlib
* Pillow (PIL)

Install dependencies using:

```
pip install torch torchvision pandas numpy scikit-learn matplotlib Pillow
```

## Setup

**Clone the Repository:**

```
git clone <repository-url>
cd <repository-directory>
```

**Prepare Data:**

* Place house images (named `1.jpg` to `17.jpg`) in an `images/` folder.
* Ensure a `data.csv` file with columns including `Lot Area`, `Overall Qual`, `SalePrice`, and optionally `image_id`.
* The script assumes 17 images; adjust `NUM_IMAGES` in the code if different.

**GPU Support:**

The script automatically uses CUDA if available; otherwise, it defaults to CPU.

## Data Preparation

* **Tabular Data**: Loaded from `data.csv` using pandas. The script selects `Lot Area` and `Overall Qual` as features and `SalePrice` as the target. Features are scaled using `StandardScaler`.
* **Image Data**: Images are loaded from the `images/` directory, resized to 224x224, and converted to tensors using `torchvision.transforms`.
* **Dataset**: A custom `HousingDataset` class combines image and tabular data for PyTorch's `DataLoader`.

## Model Architecture

The `MultimodalNet` model consists of:

* **CNN Branch**: ResNet18 (pre-trained on ImageNet) with the final fully connected layer removed, outputting 512 features.
* **Tabular Branch**: A two-layer MLP (input → 64 → 32 neurons) processing tabular features.
* **Combined Layer**: Concatenates CNN and MLP outputs (512 + 32 features), feeding into a final MLP (128 → 1) for price prediction.

## Training

**Hyperparameters:**

* Batch Size: 4
* Epochs: 5
* Learning Rate: 0.001
* Optimizer: Adam
* Loss Function: Mean Squared Error (MSE)

The model trains on 80% of the data, with progress logged per epoch.

## Evaluation

The model is evaluated on a 20% test split using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

Results are printed to the console.

## Visualization

A scatter plot compares true vs. predicted house prices, saved via `matplotlib`.

## Usage

1. Ensure the `images/` folder and `data.csv` are correctly set up.
2. Run the script:

```
python multimodal_housing.py
```

The script will:

* Load and preprocess data.
* Train the model for 5 epochs.
* Evaluate and print MAE and RMSE.
* Save the trained model as `multimodal_model.pth`.
* Display a plot of true vs. predicted prices.

## File Structure

```
project_directory/
├── images/                 # Folder containing house images (1.jpg to 17.jpg)
├── data.csv               # CSV file with tabular data
├── multimodal_housing.py  # Main script
├── multimodal_model.pth   # Saved model weights (generated after training)
└── README.md              # This file
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-branch`).
3. Commit changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

