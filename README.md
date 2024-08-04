# Iris k-NN Classifier

A k-Nearest Neighbors (k-NN) Classifier for the Iris Flower Dataset implemented in Python using NumPy and SciPy.

## Description

This project calculates the Euclidean distances between new iris samples and training samples, finds the k nearest neighbors, and predicts the types of the new samples. The accuracy of the predictions is then evaluated against known labels.

## Dataset

The dataset includes:
- `irises.npy`: Training samples with dimensions (n, 4).
- `types.npy`: Types (labels) for the training samples.
- `new_irises.npy`: New samples to be classified with dimensions (m, 4).
- `new_types.npy`: True types (labels) for the new samples to evaluate accuracy.

## Usage

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/iris-knn-classifier.git
