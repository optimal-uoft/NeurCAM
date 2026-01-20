# NeurCAM

This is the official implementation of the Neural Clustering Additive Model (NeurCAM): https://arxiv.org/abs/2408.13361

## Installation

### Using uv (Recommended)

First, install [uv](https://github.com/astral-sh/uv) if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, install NeurCAM:

```bash
# Clone the repository
git clone https://github.com/alexwolson/NeurCAM.git
cd NeurCAM

# Install the package with uv
uv pip install -e .

# For development with optional dependencies
uv pip install -e ".[dev]"
```

### Using pip

```bash
pip install -e .
```

### PyTorch with CUDA Support

If you need CUDA support for GPU acceleration, install PyTorch with CUDA separately:

```bash
# For CUDA 11.8
uv pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Quick Start

```python
from sklearn.datasets import load_iris
from neurcam import NeurCAM

# Load data
iris = load_iris()
X = iris.data

# Create and fit model
nc = NeurCAM(k=3, epochs=5000)
nc = nc.fit(X)

# Make predictions
neurcam_pred = nc.predict(X)
```

## Development

To set up the development environment:

```bash
# Install with development dependencies
uv pip install -e ".[dev]"

# Format code with black
black neurcam/
```
