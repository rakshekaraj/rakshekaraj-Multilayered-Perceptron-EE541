# MLP Network Playground

This project is a simple but flexible playground for understanding how Multilayer Perceptrons (MLPs) behave under different training setups. It uses PyTorch to build a fully connected feedforward neural network and allows experimentation with various hyperparameters, activation functions, and regularization techniques.

The goal is not to achieve state-of-the-art performance, but to better understand the dynamics of training deep (and shallow) MLPs in a controlled environment.

## What This Project Covers

- Implementation of a basic MLP from scratch using PyTorch
- Support for variable number of layers and hidden units
- Choice of activation functions (ReLU, Tanh, etc.)
- Optional dropout and weight decay
- Training and evaluation loop with performance tracking
- Visualization of loss and accuracy across epochs

This makes it a good tool for learning, debugging, or testing optimization strategies on small datasets.

## Files

- `MLP Network Playground.ipynb`: Interactive notebook for exploring training behaviors
- `MLP.py`: Modular class-based implementation of the MLP model, useful for reuse and extension
- `README.md`: Project overview and instructions

## Requirements

- Python 3.7 or higher
- PyTorch
- Matplotlib
- NumPy

Install dependencies with:

```bash
pip install torch matplotlib numpy
