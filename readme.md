# MLOpsDemo

A tiny example library API (LinearNet) built with PyTorch.

## Motivation

This project serves as a concise, practical example to illustrate core MLOps concepts

- CI/CD pipelines
- Packaging a Python package
- How to build a simple linear neural network using PyTorch

It is **designed** to provide a clear reference for those seeking to understand or replicate a minimal MLOps workflow and model structure in PyTorch.

Follow the structure of this project by going through the code and README and the reader should have a solid understanding of MLOps and how to build a simple linear neural network using PyTorch.

## Install (from TestPyPI after release) - If you would like to try it out

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple MLOpsDemo
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
- **Version**: 0.2.3
- **Author**: Tiexing Wang
- **License**: MIT

MLOpsDemo provides:

- A simple linear neural network (`LinearNet`)
- A convenience API for quick training (`fit_predict`)
- Example tests and packaging configuration

---

## Installation

Python 3.9+ is required. When published to TestPyPI or PyPI, use the command above.

Install a specific version from TestPyPI (recommended to avoid ambiguity):

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple MLOpsDemo==0.2.3
```

### Install for local development (recommended)

```bash
python -m pip install --upgrade pip
pip install -e .[dev]
```

Then run tests and basic quality checks:

```bash
pytest -q # this will run all the tests in the tests/test_api.py file

# What is a linter?
# A linter is a tool that checks your code for style issues, syntax errors, and potential bugs.
# It does not run your code, but instead "reads" your code to help you write cleaner and more maintainable programs.

# The linter will NOT run automatically unless you set it up to do so (for example, in a CI workflow).
# You need to run it manually with a command like the one below:

ruff check src tests  # This will run the linter on the src and tests directories

black . --check # this will check if the code is formatted correctly

isort . --check # this will check if the imports are sorted correctly
```

---

## What this repo includes (What I've done)

### pyproject.toml

- Packaging is managed using `pyproject.toml`, with `hatchling` as the build backend. The source code is located in the `src/MLOpsDemo` directory.

  - `hatchling` is a modern Python package build tool. Its main job is to package your project into formats like wheel or sdist. It is commonly used as the `[build-system]` backend in `pyproject.toml`.

  - `hatch` is a more complete project management tool that includes `hatchling` for building, but also helps you manage virtual environments, dependencies, versioning, project templates, and more.

  - In short: `hatch` = project management + building, while `hatchling` = just building.

- Example (`pyproject.toml` snippet):

  ```
  [build-system]
  requires = ["hatchling"]
  build-backend = "hatchling.build"
  ```

- Minimal PyTorch example API:

  - `LinearNet` (single-layer linear model)
  - `Trainer` (SGD + MSE, simple training loop)
  - `fit_predict` (one-call train and predict)

- Tests in `tests/test_api.py` (shape/accuracy sanity checks)

---

### What is the CI workflow `.github/workflows/ci.yml`?

This is an automated script (called a workflow) that lives in your project's `.github/workflows/` folder. Its main job is to help you automatically lint and test your code on every push and pull request. This ensures code quality and correctness before merging changes.

Here’s what this workflow does:

- **Python 3.10**: The workflow sets up Python version 3.10, which is the version your project uses.
- **Install `-e .[dev]`**: This command installs your project in "editable" mode along with all development dependencies (like testing and code checking tools). The `-e` means you can edit your code and the changes will be picked up right away.
- **`pytest`**: `pytest` is a popular Python testing tool. It automatically finds and runs your test code to check if your functions and classes work correctly. For example, if you write a function, `pytest` can help you make sure it returns the right results.
- **`ruff`**: `ruff` is a fast Python code linter. A linter is a tool that checks your code for style issues, formatting problems, or potential bugs—like unused imports or inconsistent naming. This helps keep your code clean and easy to read.

**Why do we do this?**

The purpose of this workflow is to automatically test your code every time you make changes. `pytest` checks if your code works, `ruff` checks if your code is clean, and the whole process helps keep your project healthy and easy to work on.

**In summary:**
The `ci.yml` workflow helps you automatically test your code for every push/PR. `pytest` checks if your code works, `ruff` checks if your code is clean, and the whole process keeps your project healthy and easy to work on.

---

### What is the publish workflow `.github/workflows/publish.yml`?

This is another automated script (called a workflow) that lives in your project's `.github/workflows/` folder. Its main job is to help you publish your Python package to [TestPyPI](https://test.pypi.org/), which is a test version of the Python Package Index (PyPI). TestPyPI lets you try out the publishing process without making your package public to everyone.

Here’s what this workflow does:

- **Build your package**: It uses tools like `build` and `twine` to create distribution files for your project. These files are usually a "wheel" (`.whl`) and a "source distribution" (`.tar.gz`). These are the formats Python uses to share and install packages.
- **Upload to TestPyPI**: After building, the workflow uploads your package files to TestPyPI using `twine`. It uses a special API token (a kind of password) stored securely in your GitHub project settings, so your real password is never exposed.
- **When does it run?**: This workflow is triggered automatically when you push a new version tag (like `v1.0.0`) to your repository. This helps make sure only official releases are published.

**Why do we do this?**

The purpose of this workflow is to automate the process of building and publishing your package. This saves you from having to run all the commands manually and helps prevent mistakes. By publishing to TestPyPI first, you can check that everything works before publishing to the real PyPI, which is public and permanent.

**In summary:**
The `publish.yml` workflow helps you automatically build and upload your Python package to TestPyPI whenever you release a new version. This makes the release process safer, faster, and less error-prone, especially as your project grows or if you work with a team.

- Publish workflow `.github/workflows/publish.yml` (TestPyPI):
  - Build `wheel` + `sdist`, upload via `twine` using `TEST_PYPI_API_TOKEN`

---

## Project Structure

```
MLOps_tutorial/
├─ src/MLOpsDemo/
│  ├─ __init__.py
│  └─ api.py
├─ tests/
│  └─ test_api.py
├─ .github/workflows/
│  ├─ ci.yml
│  └─ publish.yml
├─ pyproject.toml
├─ readme.md
└─ LICENSE
```

---

## Pre-commit (optional but recommended)

Set up local checks so you catch issues before pushing:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Sample `.pre-commit-config.yaml` you can add at repo root:

```
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: ["--fix"]
      - id: ruff-format
  - repo: https://github.com/psf/black
    rev: 24.8.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
```

---

## Contributing

1. Create a feature branch from `main` (e.g., `feature/my-change`).
2. Install dev deps: `pip install -e .[dev]`.
3. Run checks locally: `ruff check src tests`, `pytest -q`.
4. Open a Pull Request; CI must pass before merge.

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## FAQ

- How do I install PyTorch? Use the official selector at `https://pytorch.org/` to pick CPU/CUDA build matching your Python.
- `pip install -e .[dev]` fails on Windows? Ensure your venv is active and `pip` is upgraded: `python -m pip install --upgrade pip`.
- `ruff/black/isort` not found? Reinstall dev extras or install individually.
- TestPyPI cannot resolve dependencies? Use dual indexes as shown above.
