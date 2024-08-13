from numpy import real
import torch 
from torch import nn

class FuzzyCMeansLoss(nn.Module):
    def __init__(self, m=1.0, return_centroids=False):
        super(FuzzyCMeansLoss, self).__init__()
        self.m = m  # hyperparameter that controls the fuzziness of the cluster
        self.return_centroids = return_centroids

    def forward(self, X, W, centroids=None):
        """
        X is the input data of shape (batch_size, n_features)
        W is the fuzzy membership matrix of shape (batch_size, cluster_size)
        centroids is the cluster centroids of shape (cluster_size, n_features)
        """
        # Raise W to the power m
        W_raised = torch.pow(W, self.m)

        # Calculate centroids if not provided
        if centroids is None:
            centroids_num = torch.sum(W_raised.unsqueeze(2) * X.unsqueeze(1), axis=0)
            centroids_den = torch.sum(W_raised, axis=0).unsqueeze(1) + 1e-8  # Adding epsilon for numerical stability
            centroids = centroids_num / centroids_den

        # Calculate distances (batch_size, cluster_size, n_features)
        distances = torch.norm(X.unsqueeze(1) - centroids, dim=2, p=2)  # Euclidean distance

        # Calculate the loss
        loss = torch.mean(torch.pow(distances, 2) * W_raised)

        if self.return_centroids:
            return loss, centroids
        else:
            return loss