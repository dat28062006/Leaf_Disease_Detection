# Leaf Disease Detection

This repository contains a deep learning pipeline for plant leaf disease detection. The model uses a hybrid architecture combining a CNN backbone (EfficientNet-B4) for feature extraction and a Vision Transformer (ViT) block for capturing global context.

## Project Structure

- `main.py`: The main training and evaluation script. Contains data loading, model definition, training loop, and testing code.
- `train.csv`: Contains the training image paths and their corresponding labels.
- `test.csv`: Contains the testing image paths and their corresponding labels.
- `requirements.txt`: List of Python dependencies.

## Model Architecture

- **CNN Backbone**: EfficientNet-B4 (pretrained), used to extract local spatial features from the input images.
- **ViT Block**: A custom Vision Transformer block with 2 layers and 8 attention heads, processing the features extracted by the CNN to learn global representations.
- **Classifier**: A linear layer mapping the features to the target disease classes.

## Installation

Install the necessary dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## Dataset

The dataset images are expected to be located in a directory relative to the `train.csv` and `test.csv` paths. The CSV files should have columns for `image` (relative path) and `plant_disease` (string label).

## Training & Evaluation

To train and evaluate the model, simply run:

```bash
python main.py
```

The script will automatically:
1. Load data from the CSV files.
2. Setup the dataset and dataloaders with data augmentation (Random Horizontal Flip, Rotation).
3. Train the hybrid model for 10 epochs.
4. Save the best checkpoint (`best.pth`) and the latest checkpoint (`checkpoints/last.pth`).
5. Evaluate on the validation set after each epoch.
6. Evaluate the final performance (Accuracy, Precision, Recall, F1-score) on the test set (`test.csv`).
