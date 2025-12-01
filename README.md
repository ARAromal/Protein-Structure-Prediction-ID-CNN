# Protein Secondary Structure Prediction using 1D Convolutional Neural Networks (CNN)

This project applies Deep Learning techniques to a classic bioinformatics problem: predicting the secondary structure (Helix, Strand, or Coil) of a protein residue based on its local amino acid sequence.

## 🔬 Project Overview

While traditional methods often use standard feedforward networks, this project utilizes a **1D Convolutional Neural Network (1D CNN)** in MATLAB. The CNN scans the protein sequence using convolutional filters to automatically detect local motifs and chemical patterns without manual feature engineering.

### Key Technical Details
* **Framework:** MATLAB Deep Learning Toolbox
* **Architecture:** 1D CNN (Convolution -> Batch Norm -> ReLU -> MaxPool)
* **Input Encoding:** 5-bit binary encoding per amino acid
* **Window Size:** 5 residues (Scanning local context)
* **Dataset Split:** 80% Training / 20% Validation

## 🧠 Model Architecture

The custom network (`trainProteinCNN.m`) consists of the following layers:
1.  **Sequence Input:** Accepts [5x5] matrix input [TimeSteps x Channels].
2.  **Conv1D Layer:** 32 filters (size 3) with Causal Padding.
3.  **MaxPooling:** Downsamples features to capture dominant motifs.
4.  **Conv1D Layer:** 64 filters (size 3) to capture higher-level features.
5.  **Global Average Pooling:** Reduces the temporal dimension.
6.  **Flatten Layer:** Prepares data for the dense layer.
7.  **Fully Connected + Softmax:** Outputs probabilities for 3 classes (H, E, C).

## 📊 Results

The model was trained for 30 epochs using the Adam optimizer.

* **Validation Accuracy:** ~52.4%
* **Evaluation:** The model was evaluated on a 20% hold-out validation set to ensure generalization.

*Note: The performance reflects the challenge of using a restricted window size (5 residues). Deeper contexts (larger windows) generally yield higher accuracy.*

## 🚀 How to Run

1.  Clone this repository.
2.  Ensure `pr_data.m` is in the directory.
3.  Run the master script in MATLAB:
    ```matlab
    run_cnn_experiment
    ```
4.  This script will automatically:
    * Preprocess the raw data.
    * Define the network architecture.
    * Train the model.
    * Save the results to `cnn_model.mat`.
