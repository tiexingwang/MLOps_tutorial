# MLOpsDemo Project Guide

This guide explains the repository structure, key files, APIs, tests, and CI/CD. Copy this file to docs/PROJECT_GUIDE.md in your repo.

## Structure

```
MLOps_tutorial/
├─ src/MLOpsDemo/
│  ├─ __init__.py
│  └─ api.py
├─ tests/test_api.py
├─ .github/workflows/ci.yml
├─ .github/workflows/publish.yml
├─ pyproject.toml
├─ readme.md
└─ docs/PROJECT_GUIDE.md
```

## pyproject.toml (packaging and deps)

- build-system
  - uses `hatchling` as the backend.
- project
  - name: `MLOpsDemo`
  - version: `0.2.2` (keep in sync with `__version__` below)
  - readme: `readme.md` (case-sensitive on CI)
  - requires-python: `>=3.9`
  - license, authors: metadata
- dependencies
  - `numpy>=1.24,<3.0`
  - `torch>=2.0,<3.0`
- optional-dependencies (dev)
  - `pytest`, `build`, `twine`, `ruff`, `black`, `pytest-cov`
- hatch wheel target
  - `packages = ["src/MLOpsDemo"]` (src-layout)

## src/MLOpsDemo/**init**.py (package entry)

```python
__all__ = ["LinearNet", "fit_predict", "Trainer"]
__version__ = "0.2.2"

from .api import LinearNet, fit_predict, Trainer
```

- `__all__`: what `from MLOpsDemo import *` would export.
- `__version__`: external version; must match `pyproject.toml`.
- re-export API symbols from `api.py` for convenient imports.

## src/MLOpsDemo/api.py (core API)

```python
import torch
import torch.nn as nn
import torch.optim as optim
```

- import PyTorch tensor ops, layers, and optimizers.

```python
class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc1 = nn.Linear(1, 1)
    def forward(self, x):
        return self.fc1(x)
```

- single-layer linear model (1D -> 1D).
- `forward` returns the linear layer output.

```python
class Trainer:
    """
    A class for training a linear model.
    """
    def __init__(self):
        self.model = LinearNet()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
```

- wraps training concerns: model, optimizer (SGD, lr=0.01), loss (MSE).

```python
    def fit(self, X, y, epochs=100):
        """
        Train the model for a given number of epochs.
        """
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            y_pred = self.model(X)
            loss = self.criterion(y_pred, y)
            loss.backward()
            self.optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")
        return loss.item()
```

- iterative training, returns final scalar loss (float).

```python
    def predict(self, X):
        """
        Predict the output for a given input.
        """
        with torch.no_grad():
            return self.model(X)
```

- inference without gradients; returns tensor predictions.

```python
def fit_predict(X, y, epochs=100):
    """
    Fit the model and predict the output for a given input.
    """
    trainer = Trainer()
    loss = trainer.fit(X, y, epochs)
    print(f"Final Loss: {loss}")
    return trainer.predict(X), loss
```

- convenience API: train then return `(pred, loss)`; matches tests.

## tests/test_api.py (unit tests)

Key checks:

- `LinearNet` structure: has `fc1`.
- single-sample fit:
  - `pred, loss = fit_predict(x, y, epochs=200)`
  - `loss < 0.1`, `pred.shape == (1,1)`, value close to target.
- multi-sample shape/type checks with more epochs.

## .github/workflows/ci.yml (CI)

- Python 3.10
- install dev deps: `pip install -e .[dev]`
- lint: `ruff check src tests` (new CLI)
- test: `pytest -q`

## .github/workflows/publish.yml (Publish to TestPyPI)

- triggers on tags starting with `v` (e.g., `v0.2.2`)
- build: `python -m build` (wheel + sdist)
- upload: `twine upload --repository-url https://test.pypi.org/legacy/ dist/*`
- requires repo secret `TEST_PYPI_API_TOKEN` (from `test.pypi.org`), username `__token__`.

Recommended improvements (optional):

- add `python -m twine check dist/*` before upload
- add `-v` to `twine upload` for clearer errors
- clean `dist/ build/ *.egg-info` before build

## README highlights

Install from TestPyPI with dual indexes so deps resolve from PyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple MLOpsDemo
```

Minimal usage:

```python
from MLOpsDemo import LinearNet, fit_predict
import torch

X = torch.tensor([[1.0]])
y = torch.tensor([[5.0]])

pred, loss = fit_predict(X, y, epochs=100)
print(pred)
print(loss)
```

## Release checklist

1. Sync versions

- `pyproject.toml [project].version`
- `src/MLOpsDemo/__init__.py __version__`

2. Local checks

- `ruff check src tests`
- `pytest -q`

3. Tag and push

```bash
git add -A && git commit -m "chore: bump version to 0.x.y"  # if needed
git tag -a v0.x.y -m "release 0.x.y"
git push && git push origin v0.x.y
```

4. Verify install (fresh env)

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple MLOpsDemo==0.x.y
python -c "import MLOpsDemo as m; print(m.__version__)"
```

## FAQ

- 400 Bad Request on upload:
  - usually re-uploading an existing version; bump patch version and rebuild.
- Metadata/readme errors on build:
  - readme path/name must match exactly (case-sensitive).
- Dependencies not found on TestPyPI:
  - use dual indexes (`--extra-index-url https://pypi.org/simple`).
- Ruff “unrecognized subcommand”:
  - use `ruff check ...` with newer versions.

```

```
