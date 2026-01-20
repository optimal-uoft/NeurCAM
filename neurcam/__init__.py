"""
NeurCAM: Neural Clustering Additive Model

A Python package for interpretable clustering using neural networks.

This package implements the Neural Clustering Additive Model (NeurCAM),
which combines neural networks with interpretable clustering to provide
both accurate clustering and model interpretability.
"""

from neurcam.loss import FuzzyCMeansLoss
from neurcam.model import NeurCAM, NeurCAMModel

__version__ = "0.1.0"
__all__ = ["NeurCAM", "NeurCAMModel", "FuzzyCMeansLoss"]
