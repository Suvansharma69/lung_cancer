# Lung Cancer Survival Prediction

This project implements a deep learning model to predict lung cancer patient survival based on various clinical and demographic features.

## Features

- Data preprocessing and feature engineering
- Neural network model with GPU support
- Model training with validation
- Performance evaluation and visualization
- Automatic model checkpointing

## Requirements

- Python 3.7+
- PyTorch
- CUDA-compatible GPU (recommended)
- Other dependencies listed in requirements.txt

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python lung_cancer_prediction.py
```

The script will:
1. Load and preprocess the data
2. Train the model using GPU if available
3. Save the best model checkpoint
4. Generate training history plot
5. Evaluate the model on the test set

## Model Architecture

The model uses a neural network with:
- Input layer
- 3 hidden layers (128, 64, 32 neurons)
- Dropout for regularization
- ReLU activation functions
- Sigmoid output for binary classification

## Data Preprocessing

The preprocessing pipeline includes:
- Date feature engineering
- Categorical variable encoding
- Numerical feature scaling
- Train/validation/test split

## Output

- `best_model.pth`: Saved model checkpoint
- `training_history.png`: Training and validation loss plot
- Console output with training progress and final accuracy 