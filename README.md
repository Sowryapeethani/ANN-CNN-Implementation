# ANN-CNN-image-classification-PyCUDA

This project demonstrates the implementation of Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN) with GPU acceleration using PyCUDA. It includes data generation, model training, and evaluation functions optimized for GPU processing.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [License](#license)

## Project Overview

This project leverages PyCUDA to accelerate neural network operations, implementing key functions for both ANN and CNN models on the GPU. It explores the integration of CUDA with Python for neural network training, aiming to provide faster computation times compared to CPU-based processing.

## Features

- Data generation for model training and testing
- Implementation of an Artificial Neural Network (ANN)
- Implementation of a Convolutional Neural Network (CNN)
- GPU-based acceleration using PyCUDA
- Evaluation metrics: accuracy score, classification report

## Installation

1. Clone the repository:
   ```bash
   git https://github.com/Pulakhandam-Amrutha/ANN-CNN-implementation-PyCuda.git
   cd ANN-CNN-implementation-PyCuda
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Install PyCUDA if not already installed:
   ```bash
   pip install pycuda
   ```

## Usage

1. Run the Jupyter notebook:
   ```bash
   jupyter notebook "ANN and CNN-implementation-PyCuda.ipynb"
   ```
2. Execute the cells sequentially to install dependencies, generate data, and run ANN and CNN models on the GPU.

## Dependencies

- Python 3.x
- PyCUDA
- NumPy
- Scikit-learn

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
