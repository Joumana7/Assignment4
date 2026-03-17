# MNIST Training with PyTorch and MLflow

This repository contains a PyTorch-based machine learning training script for the MNIST dataset with experiment tracking using MLflow.

The project trains a simple Multi-Layer Perceptron (MLP) on handwritten digit images from MNIST, evaluates the model on validation and test sets, and logs parameters, metrics, and the trained model to MLflow.

---

## Project Overview

This project performs the following tasks:

- Downloads and preprocesses the MNIST dataset
- Splits the training data into training and validation sets
- Builds a simple MLP model using PyTorch
- Trains the model for a specified number of epochs
- Evaluates the model on validation and test datasets
- Logs parameters, metrics, and model artifacts using MLflow

---

## Model Architecture

The model used in this project is a simple feedforward neural network.

Architecture:

- Flatten input image (28x28)
- Linear layer: 784 → 256
- ReLU activation
- Linear layer: 256 → 128
- ReLU activation
- Linear layer: 128 → 10 (number of digit classes)

This model is implemented in the `SimpleMLP` class.

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- MLflow

---

## Project Structure

Typical repository structure:

```
project-root/
│
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml
│
├── requirements.txt
├── README.md
├── train.py
└── other project files
```

---

## Installation

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

---

## How to Run the Training Script

Run the training script from the terminal:

```bash
python train.py 0.01 3 64 42 Joumana 202202100 ./mlruns ./data
```

### Script Arguments

The script expects the following positional arguments:

1. learning_rate → learning rate for optimizer  
2. epochs → number of training epochs  
3. batch_size → batch size used for training  
4. seed → random seed for reproducibility  
5. name → student name  
6. student_id → student ID  
7. tracking_dir → directory where MLflow stores experiment logs  
8. data_dir → directory where the MNIST dataset will be stored

Example:

```bash
python train.py 0.001 5 32 42 Joumana 202202100 ./mlruns ./data
```

---

## Dataset

This project uses the MNIST dataset from `torchvision.datasets.MNIST`.

The dataset contains handwritten digit images (0–9).

The script automatically downloads:

- Training dataset
- Test dataset

The training dataset is split into:

- 90% training data
- 10% validation data

---

## Training Process

During training the script:

- Loads MNIST data using PyTorch DataLoader
- Trains the model using Stochastic Gradient Descent (SGD) with momentum
- Calculates training loss and accuracy
- Evaluates validation loss and accuracy after each epoch
- Prints results for each epoch

After training, the model is evaluated on the test dataset.

Example console output:

```
Epoch 1/3 | train_loss=... train_acc=... val_loss=... val_acc=...
Epoch 2/3 | train_loss=... train_acc=... val_loss=... val_acc=...
Epoch 3/3 | train_loss=... train_acc=... val_loss=... val_acc=...
Final test_loss=... test_accuracy=...
```

---

## MLflow Experiment Tracking

This project uses MLflow to track machine learning experiments.

The following information is logged:

### Parameters
- learning rate
- number of epochs
- batch size
- optimizer type
- random seed
- model name
- dataset name

### Tags
- student ID
- course name

### Metrics
- training loss
- training accuracy
- validation loss
- validation accuracy
- test loss
- test accuracy

### Model Artifact
The trained model is saved using:

```python
mlflow.pytorch.log_model(model, artifact_path="model")
```

---

## GitHub Actions CI Pipeline

This repository also includes a GitHub Actions workflow to automatically validate the project environment.

The CI pipeline performs the following steps:

1. Checkout the repository
2. Setup Python 3.10
3. Install dependencies from `requirements.txt`
4. Run a linter check
5. Run a dry test to confirm the ML environment is ready
6. Upload project documentation as an artifact

---

## Workflow Trigger

The GitHub Actions workflow runs automatically when:

- Code is pushed to any branch except `main`
- A pull request is created

---

## Artifact

The pipeline uploads the following file as a GitHub artifact:

```
README.md
```

Artifact name:

```
project-doc
```

---

## Notes

This repository is part of an MLOps assignment focused on implementing automated validation using GitHub Actions.

---

## Author

Joumana  
Student ID: 202202100
