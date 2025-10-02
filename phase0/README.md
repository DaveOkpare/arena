# Phase 0: MNIST & Fashion-MNIST Classification

A simple 3-layer MLP (Multi-Layer Perceptron) trained on MNIST and Fashion-MNIST datasets with a reproducible training pipeline.

## Results

- **MNIST**: 97.9% test accuracy, 0.084 test loss
- **Fashion-MNIST**: 88.3% test accuracy, 0.324 test loss

## Architecture

**Pyramid MLP**: 784 → 512 → 256 → 10
- Input layer: 784 (28×28 flattened images)
- Hidden layer 1: 512 neurons + ReLU
- Hidden layer 2: 256 neurons + ReLU
- Output layer: 10 classes

## Key Changes That Improved Performance

### 1. Pyramid Architecture (784→512→256→10)
**Why it worked:**
- Forces hierarchical feature learning: pixels → edges → patterns → classes
- Each layer compresses information, preventing memorization
- Fewer parameters (131k vs 262k in middle layer) = better generalization
- Matches natural feature hierarchy in vision tasks

### 2. Adam Optimizer with lr=1e-3
**Why it worked:**
- Adam has adaptive learning rates per parameter
- Combines momentum + RMSprop benefits
- Handles lr=1e-3 well (unlike vanilla SGD which needed 1e-4)
- Faster convergence with built-in stability

### 3. Optimizer Matters More Than You Think
**Learning**:
- SGD with lr=1e-3 was too aggressive → oscillated
- Adam with lr=1e-3 worked perfectly → adaptive step sizes
- **Takeaway**: Optimizer choice is as important as learning rate

## Reproducibility

All experiments are fully reproducible with:
- Fixed random seed (42)
- Seed set for: Python, NumPy, PyTorch, MPS/CUDA
- Deterministic CUDA operations enabled

## Usage

```bash
# Train model
make train

# Evaluate model
make eval
```

## Configuration

All hyperparameters centralized in `config.py`:
- Seed: 42
- Batch size: 64
- Learning rate: 1e-3
- Epochs: 5
- Hidden sizes: 512, 256

## Project Structure

```
phase0/
├── config.py       # All hyperparameters
├── model.py        # MLP architecture
├── main.py         # Training script
├── eval.py         # Evaluation script
├── utils.py        # Seed setting utilities
├── Makefile        # Train/eval commands
└── README.md       # This file
```

## Requirements

- torch
- torchvision
- numpy

## Training Details

- Loss function: CrossEntropyLoss
- Optimizer: Adam
- No regularization (dropout, weight decay)
- No data augmentation
- No learning rate scheduling
- No early stopping

## Analysis

Both datasets show healthy train/test gaps:
- **MNIST**: Train loss ~0.032, Test loss 0.084 (minimal overfitting)
- **Fashion-MNIST**: Train loss 0.351, Test loss 0.324 (slight underfitting)

Fashion-MNIST has room for improvement with:
- Learning rate scheduling
- Early stopping
- Gradient clipping
- Data augmentation
