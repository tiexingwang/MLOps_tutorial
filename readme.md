# MLOpsDemo

A tiny example library API (linear net) built with PyTorch.

## Install (from TestPyPI after release)

```bash
pip install --index-url https://test.pypi.org/simple/ mlopsdemo
```

## Usage

```python
from MLOpsDemo import LinearNet, fit_predict
import torch

# Training data: y ≈ 2x + 3
X = torch.tensor([[1.0]])
y = torch.tensor([[5.0]])

# Quick training; returns a scalar loss (see tests)
pred, loss = fit_predict(X, y, epochs=100)
print(pred)  # e.g., 0.05
print(loss)  # e.g., 0.05

# Optional: use the raw model forward (untrained in this snippet)
model = LinearNet()
with torch.no_grad():
    print(model(torch.tensor([[4.0]])))
```

---

## Project Overview

- **Project Name**: MLOpsDemo
- **Version**: 0.2.0
- **Author**: Tiexing Wang
- **License**: MIT

MLOpsDemo provides:

- A simple linear neural network (`LinearNet`)
- A convenience API for quick training (`fit_predict`)
- Example tests and packaging configuration

---

## Installation

Python 3.9+ is required. When published to TestPyPI or PyPI, use the command above.
