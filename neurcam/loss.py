"""Loss functions for NeurCAM clustering."""

import torch
from torch import nn
from typing import Optional, Union


class FuzzyCMeansLoss(nn.Module):
    """
    Fuzzy C-Means loss function for soft clustering.

    This loss implements the objective function of fuzzy c-means clustering,
    which minimizes the weighted sum of squared distances to cluster centroids.
    """

    def __init__(self, m: float = 1.0, return_centroids: bool = False) -> None:
        """
        Initialize the Fuzzy C-Means loss.

        Args:
            m: Fuzziness parameter that controls the degree of cluster overlap.
               Higher values lead to fuzzier clusters. Must be >= 1.0.
            return_centroids: If True, return both loss and centroids.
        """
        super(FuzzyCMeansLoss, self).__init__()
        if m < 1.0:
            raise ValueError(f"Fuzziness parameter m must be >= 1.0, got {m}")
        self.m = m
        self.return_centroids = return_centroids

    def forward(
        self,
        X: torch.Tensor,
        W: torch.Tensor,
        centroids: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute the fuzzy c-means loss.

        Args:
            X: Input data of shape (batch_size, n_features).
            W: Fuzzy membership matrix of shape (batch_size, n_clusters).
            centroids: Cluster centroids of shape (n_clusters, n_features).
                      If None, centroids are computed from X and W.

        Returns:
            Loss value, or tuple of (loss, centroids) if return_centroids is True.
        """
        # Raise W to the power m for fuzzy weighting
        W_raised = torch.pow(W, self.m)

        # Calculate centroids if not provided
        if centroids is None:
            centroids_num = torch.sum(W_raised.unsqueeze(2) * X.unsqueeze(1), axis=0)
            centroids_den = torch.sum(W_raised, axis=0).unsqueeze(1) + 1e-8
            centroids = centroids_num / centroids_den

        # Calculate Euclidean distances: (batch_size, n_clusters)
        distances = torch.cdist(X, centroids, p=2)

        # Calculate the loss: mean of squared distances weighted by membership
        loss = torch.mean(torch.pow(distances, 2) * W_raised)

        if self.return_centroids:
            return loss, centroids
        else:
            return loss
