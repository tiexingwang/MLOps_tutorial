# --------------------------------
# test_api.py
# This file contains the tests for the api.py file.
# define the x and y for the linear net
# x = torch.randn(100, 1)
# y = 2 * x + 1
# --------------------------------

import pytest
from MLOpsDemo.api import LinearNet, fit_predict
import torch

def test_linear_net():
    model = LinearNet()
    assert model is not None
    assert model.fc1 is not None

def test_fit_predict():
    # Define x and y as single numbers (1 sample)
    x = torch.tensor([[2.0]])
    y = torch.tensor([[5.0]])
    pred, loss = fit_predict(x, y, epochs=200)
    # Check that loss is a single number and is small
    assert isinstance(loss, float) or isinstance(loss, torch.Tensor)
    assert loss < 0.1
    assert pred is not None
    assert pred.shape == (1, 1)
    assert pred[0, 0] - 5.0 < 1e-2

def test_fit_predict_multiple_samples():
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = torch.tensor([[3.0], [5.0], [6.0], [8.0], [10.0]])
    pred, loss = fit_predict(x, y, epochs=500)
    assert isinstance(loss, float) or isinstance(loss, torch.Tensor)
    assert pred is not None
    assert pred.shape == (5, 1)




