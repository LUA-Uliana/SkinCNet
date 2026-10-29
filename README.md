# SkinCNet: Compact Multimodal Neural Network Optimized via Knowledge Distillation for Skin Lesion Classification

## Description

This repository contains the official implementation of the study on a compact multimodal neural network SkinCNet optimized via knowledge distillation for pigmented skin lesion classification. The framework jointly processes dermatoscopic images and clinical patient metadata (age group, sex, anatomical localization) and transfers knowledge from heavy teacher architectures (DenseNet_201, ResNeXt_101, Inception_v4) to the lightweight student model SkinCNet.

## Software Environment

| Parameter | Value |
| --- | --- |
| Operating System | Windows 11 |
| Python | 3.11.11 |
| PyTorch | 2.x |
| CUDA | 12.1 |
| GPU | NVIDIA RTX 4070 Ti SUPER |

## Installation

Clone the repository:

    git clone https://github.com/Lyakhova-Uliana/SkinCNet-Compact-Multimodal-Neural-Network-Optimized-via-Knowledge-Distillation-.git
    cd SkinCNet-Compact-Multimodal-Neural-Network-Optimized-via-Knowledge-Distillation-

Create a Conda environment:

    conda env create -f environment.yml
    conda activate NN_3_11_11

Or install dependencies via pip:

    pip install -r requirements.txt

## Repository Structure

    .
    ├── MNN based on DenseNet_201 as a model teacher.ipynb
    ├── MNN based on DenseNet_201 as a model teacher test/
    ├── MNN based on Inception_v4 as a model teacher.ipynb
    ├── MNN based on Inception_v4 as a model teacher test/
    ├── MNN based on ResNeXt_101 as a model teacher.ipynb
    ├── MNN based on ResNeXt_101 as a model teacher test/
    ├── MNN based on specialized CNN/
    ├── Ablation Study SkinCNet.ipynb
    ├── requirements.txt
    ├── environment.yml
    ├── LICENSE
    └── README.md

## Reproducing the Experiments

Step 1. Train teacher models. Open the corresponding notebooks and execute all cells sequentially:

    MNN based on DenseNet_201 as a model teacher.ipynb
    MNN based on ResNeXt_101 as a model teacher.ipynb
    MNN based on Inception_v4 as a model teacher.ipynb

Each notebook contains data loading, hair removal with the RemovalHS method, metadata encoding, training loop with weighted cross-entropy loss, and statistical evaluation over 10 independent runs.

Step 2. Train the SkinCNet student model. Open the directory "MNN based on specialized CNN" and execute the notebooks. This stage includes training without knowledge distillation (baseline) and training with knowledge distillation from each teacher model with different temperature values T = 1, 2, 5 and weight parameter values 0.25, 0.5, 0.75.

Step 3. Ablation study. Open the notebook "Ablation Study SkinCNet.ipynb". This notebook contains comparison of 7 convolutional architectures, evaluation of fusion strategies (Concatenation, SE, Cross-Attention, Gated), analysis of metadata encoding methods, and evaluation of hair removal impact.

Step 4. External validation on PAD-UFES-20. The validation uses a stratified 60/20/20 train/validation/test split based on unique patient IDs. Models were fine-tuned for 5 epochs on the training subset before evaluation.

## Hyperparameters

| Parameter | Value |
| --- | --- |
| Maximum epochs | 50 |
| Optimizer | SGD with momentum 0.9 |
| Initial learning rate | 0.01 |
| Batch size | 16 |
| Independent runs | 10 |
| Loss function | Weighted Cross-Entropy |
| Knowledge distillation temperature | 1, 2, 5 |
| Knowledge distillation weight | 0.25, 0.5, 0.75 |
| Input resolution (SkinCNet, DenseNet, ResNeXt) | 224 x 224 |
| Input resolution (Inception_v4) | 299 x 299 |

## Citation

If you use this code in your research, please cite the corresponding publication in IEEE Transactions on Neural Networks and Learning Systems.

## License

This project is distributed under the MIT License.
