# Fisher's Linear Discriminant Analysis (LDA) Implementation

## Overview
This project demonstrates the implementation of Fisher's Linear Discriminant Analysis (LDA) for two-class classification using Python and NumPy. The implementation provides a comprehensive analysis of class separation and optimal projection.

## Problem Statement
Given two classes C1 and C2 with the following samples:
- Class C1: `[[1, 2], [2, 3], [3, 3], [4, 5], [5, 5]]`
- Class C2: `[[1, 0], [2, 1], [3, 1], [3, 2], [5, 3], [6, 5]]`

The goal is to find the optimal linear transformation that maximizes class separability.

## Features
- Compute class means and scatter matrices
- Calculate optimal projection direction
- Visualize data in original and transformed spaces
- Compute Fisher criterion and class separability metrics

## Mathematical Approach
Fisher's LDA aims to find a projection vector w that:
- Maximizes between-class variance
- Minimizes within-class variance
- Defined by the equation: w ∝ Sw^-1 * (μ1 - μ2)

## Requirements
- Python 3.8+
- NumPy
- Matplotlib

## Installation
```bash
git clone https://github.com/yourusername/fishers-lda-analysis.git
cd fishers-lda-analysis
pip install numpy matplotlib
```

## Usage
```python
from fishers_lda import compute_fisher_lda

# Define your classes
C1 = [[1, 2], [2, 3], [3, 3], [4, 5], [5, 5]]
C2 = [[1, 0], [2, 1], [3, 1], [3, 2], [5, 3], [6, 5]]

# Compute LDA
results = compute_fisher_lda(C1, C2)
```

## Key Metrics Computed
- Optimal Projection Direction
- Fisher Criterion
- Between-Class Variance
- Within-Class Variance
- Class Separability

## Visualization
The implementation generates two plots:
1. Original data space with two classes
2. Projected data onto the discriminant direction

## Mathematical Foundations
- Within-Class Scatter Matrix (Sw)
- Between-Class Scatter Matrix (Sb)
- Generalized Eigenvalue Problem
- Optimal Linear Transformation

## Acknowledgments
- Inspired by the CS 5710 Intro Machine Learning course
- Fisher's Linear Discriminant Analysis original paper

---

**Note**: This is an educational implementation for machine learning practice.
