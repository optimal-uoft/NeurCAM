# NeurCAM
This is the official implementation of the Neural Clustering Additive Model (NeurCAM): https://arxiv.org/abs/2408.13361

## Environment Setup
To set up the environment, run the following commands:

```
conda env create -f environment.yml
conda activate NeurCAM
```

## Example
```
from sklearn.datasets import load_iris
from NeurCAM import NeurCAM

iris = load_iris()
X = iris.data

nc = NeurCAM(k=3, epochs=5000)
nc = nc.fit(X)
neurcam_pred = nc.predict(X)
```
