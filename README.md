# Protein-Structure-Prediction-ID-CNN
This project implements a Deep Learning approach to predict protein secondary structure (Helix, Strand, or Coil) from amino acid sequences. It utilizes a 1D Convolutional Neural Network (1D CNN) implemented in MATLAB.
# Protein Secondary Structure Prediction using 1D CNN

This project implements a Deep Learning approach to predict protein secondary structure (Helix, Strand, or Coil) from amino acid sequences. It utilizes a **1D Convolutional Neural Network (1D CNN)** implemented in MATLAB.

## Project Overview

Unlike standard feedforward networks, this 1D CNN scans the protein sequence using convolutional filters to automatically detect local motifs and patterns.

* **Window Size:** 5 residues (Configurable)
* **Encoding:** 5-bit binary encoding per amino acid
* **Data Split:** 80% Training, 20% Validation

## Network Architecture

The model architecture (`trainProteinCNN.m`) consists of:
1.  **Sequence Input:** Accepts [5x5] matrix input.
2.  **Conv1D Layer:** 32 filters (size 3) + BatchNorm + ReLU.
3.  **MaxPooling:** Downsampling by factor of 2.
4.  **Conv1D Layer:** 64 filters (size 3) + BatchNorm + ReLU.
5.  **Global Average Pooling:** To reduce feature dimensions.
6.  **Flatten Layer:** Prepares data for classification.
7.  **Fully Connected + Softmax:** Output to 3 classes (H, E, C).

## Results

The model was trained for 30 epochs using the Adam optimizer.

* **Validation Accuracy:** ~52.4%
* **Training Plot:**
    ![Training Progress](training_progress_cnn.png)

## How to Run

1.  Ensure you have `pr_data.m` in the folder.
2.  Run the master script:
    ```matlab
    run_cnn_experiment
    ```
3.  This script will automatically:
    * Prepare the data.
    * Train the CNN.
    * Save the results to `cnn_model.mat`.
