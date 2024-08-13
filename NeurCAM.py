import pandas as pd
import numpy as np
import os 
import gc
from tqdm import trange

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import MiniBatchKMeans, KMeans
import random

from entmax import Entmax15
from Loss import FuzzyCMeansLoss


class NeurCAM:
    def __init__(self,
                 k,
                 random_state:int = 42,
                 m: float=1.05,
                 hidden_layers:list[int] = [128,128],
                 n_bases:int = 64,
                 learning_rate: float = 2e-3,
                 epochs: int = 5000,
                 batch_size: int = 512,
                 single_feature_channels: float | int = 1.0,
                 pairwise_feature_channels: float | int = 0.0,
                 warmup_ratio: float | int = 0.4,
                 o1_anneal_ratio: float | int = 0.1,
                 o2_anneal_ratio: float | int = 0.1,
                 min_temp: float =1e-5,
                 kl_weight:float  = 1.0,
                 smart_init: str = 'none',
                 model_dir: str = 'NeurCAMCheckpoints',
                 verbose = True
                 ):
        """
        NeurCAM class for interpretable clustering.

        Attributes:
            k (int): Number of clusters.
            random_state (int): Random seed for reproducibility.
            m (float): Fuzziness parameter.
            hidden_dim (int): Dimension of the hidden layer for the backbone.
            n_bases (int): output dimension of the backbone.
            learning_rate (float): Learning rate for the optimizer.
            epochs (int): Total number of training epochs.
            batch_size (int): Batch size for training.
            single_feature_channels (float | int): Number of channels for single feature interactions. If values are <=1.0, it is interpreted as a ratio of the number of features. If values are >1.0, it is interpreted as the number of channels. (set to 1.01 for one channel per feature)
            pairwise_feature_channels (float | int): Number of channels for pairwise feature interactions.
            warmup_ratio (float | int): Ratio of warmup epochs.
            o1_anneal_ratio (float | int): Ratio of first annealing phase.
            o2_anneal_ratio (float | int): Ratio of second annealing phase.
            min_temp (float): Minimum temperature for annealing.
            kl_weight (float): Weight for the KL divergence loss.
            smart_init (str): Clustering initialization method.
            model_dir (str): Directory to save model checkpoints.
        """
        self.k = k
        self.random_state = random_state
        self.m = m
        self.hidden_layers = hidden_layers
        self.n_bases = n_bases
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.sf_channels = single_feature_channels
        self.pf_channels = pairwise_feature_channels
        self.warmup_ratio = warmup_ratio
        self.o1_anneal_ratio = o1_anneal_ratio
        self.o2_anneal_ratio = o2_anneal_ratio
        self.min_temp = min_temp
        self.kl_weight = kl_weight
        self.smart_init = smart_init
        self.model_dir = model_dir
        self.model = None
        self.verbose = verbose

        self.warmup_epochs = int(self.epochs * self.warmup_ratio)
        self.o1_anneal_epochs = int(self.epochs * self.o1_anneal_ratio)
        self.o2_anneal_epochs = int(self.epochs * self.o2_anneal_ratio)
        self.feature_names = None
        self.n_features = -1
        self.repr_dim = -1

    def fit(self, X: pd.DataFrame| np.ndarray, X_repr: pd.DataFrame | np.ndarray = None):
        """
        Fit the NeurCAM model.
        Args:
            X (pd.DataFrame | np.ndarray): Input data in the interpretable space. 
            X_repr (pd.DataFrame | np.ndarray): Input data in transformed/latent space (optional).
        """
        # seed random state
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        random.seed(self.random_state)

        # create model dir
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns
            X = X.values
        if X_repr is not None:
            if isinstance(X_repr, pd.DataFrame):
                X_repr = X_repr.values
        else:
            X_repr = X

        self.n_features = X.shape[1]
        self.repr_dim = X_repr.shape[1]
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(X_repr, dtype=torch.float32))
        dataloader =  DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        if self.sf_channels <= 1.0:
            self.single_feature_channels = max(0, int(self.sf_channels * self.n_features))
        else:
            self.single_feature_channels = int(self.sf_channels)
        if self.pf_channels <= 1.0:
            self.pairwise_feature_channels = max(0, int(self.pf_channels * self.n_features))
        else:
            self.pairwise_feature_channels = int(self.pf_channels)

        model = NeurCAMModel(
            input_dim = self.n_features,
            repr_dim = self.repr_dim,
            o1_channels = self.single_feature_channels,
            o2_channels = self.pairwise_feature_channels,
            n_bases = self.n_bases,
            hidden_layers = self.hidden_layers,
            n_clusters = self.k
        )
        

        if self.smart_init == 'kmeans':
            kmeans = KMeans(n_clusters=self.k, random_state=self.random_state, n_init=10)
            kmeans.fit(X_repr)
            model.centroids.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

        elif self.smart_init == 'mbkmeans':
            kmeans = MiniBatchKMeans(n_clusters=self.k, random_state=self.random_state, n_init=10)
            kmeans.fit(X_repr)
            model.centroids.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        else:
            model._initialize_centroids(dataloader, init_size=3)
            

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=False)
        loss_func = FuzzyCMeansLoss(m=self.m)
        kld_loss = nn.KLDivLoss(reduction='batchmean')

        if self.verbose:
            print('Starting warmup phase...')
            p1_bar = trange(self.warmup_epochs, desc='Warmup Phase')
        else:
            p1_bar = range(self.warmup_epochs)
            
        best_loss = np.inf
        best_ckpt = None
        model.train()
        epoch_losses = []
        best_epoch = 0
        for epoch in p1_bar:
            model.train()
            epoch_loss = {
                'epoch': epoch,
                'clust_loss': 0.0,
                'kl_div': 0.0,
            }
            n_points = 0
            for batch in dataloader:
                optimizer.zero_grad()
                x, x_repr = batch
                network_result = model(x)
                assignments = network_result['assignments']
                clust_loss = loss_func(x_repr, assignments, centroids=model.centroids)
                loss = clust_loss
                loss.backward()
                optimizer.step()
                scheduler.step(loss)
                epoch_loss['clust_loss'] += clust_loss.item()
                n_points += x.shape[0]
                p1_bar.set_postfix({'clust_loss': loss.item()})
               
            epoch_loss['clust_loss'] /= n_points
            epoch_losses.append(epoch_loss)
            if epoch_loss['clust_loss'] < best_loss:
                best_loss = epoch_loss['clust_loss']
                best_ckpt = model.state_dict()
                best_epoch = epoch
            if epoch - best_epoch > 100:
                break
        
        model.load_state_dict(best_ckpt)
        model_copy = NeurCAMModel(
            input_dim = self.n_features,
            repr_dim = self.repr_dim,
            o1_channels = self.single_feature_channels,
            o2_channels = self.pairwise_feature_channels,
            n_bases = self.n_bases,
            hidden_layers = self.hidden_layers,
            n_clusters = self.k
        )
        model_copy.load_state_dict(best_ckpt)
        model_copy.eval()

        
        
        if self.pairwise_feature_channels > 0:
            if self.verbose and self.pairwise_feature_channels > 0:
                print('Starting pairwise shape function annealing...')
                if self.verbose:
                    p2_bar = trange(self.o2_anneal_epochs, desc='O2 Annealing Phase')
                else:
                    p2_bar = range(self.o2_anneal_epochs)

            for epoch in p2_bar:
                model.train()
                model._anneal_o2(epoch, self.o2_anneal_epochs, self.min_temp)
                valid_o2 = model._o2_valid_cuts()
                if valid_o2:
                    break
                epoch_loss = {
                    'epoch': epoch,
                    'clust_loss': 0.0,
                    'kl_div': 0.0,
                }
                n_points = 0
                for batch in dataloader:
                    optimizer.zero_grad()
                    x, x_repr = batch
                    network_result = model(x)
                    assignments = network_result['assignments']
                    log_assignments = network_result['log_assignments']
                    old_assignments = model_copy(x)['assignments']

                    clust_loss = loss_func(x_repr, assignments, centroids=model.centroids)
                    kl_div = kld_loss(log_assignments, old_assignments) * self.kl_weight
                    loss = kl_div + clust_loss

                    loss.backward()
                    optimizer.step()
                    scheduler.step(loss)
                    epoch_loss['clust_loss'] += clust_loss.item()
                    epoch_loss['kl_div'] += kl_div.item()
                    n_points += x.shape[0]
                    p2_bar.set_postfix({'clust_loss': loss.item()})
                
                epoch_loss['clust_loss'] /= n_points
                epoch_loss['kl_div'] /= n_points
                epoch_losses.append(epoch_loss)
            model._lock_in_o2()
        
        if self.verbose and self.single_feature_channels > 0:
            print('Starting single feature shape function annealing...')
            if self.verbose:
                p3_bar = trange(self.o1_anneal_epochs, desc='O1 Annealing Phase')
            else:
                p3_bar = range(self.o1_anneal_epochs)

        if self.single_feature_channels > 0:
            for epoch in p3_bar:
                model.train()
                model._anneal_o1(epoch, self.o1_anneal_epochs, self.min_temp)
                valid_o1 = model._o1_valid_cuts()
                if valid_o1:
                    break
                epoch_loss = {
                    'epoch': epoch,
                    'clust_loss': 0.0,
                    'kl_div': 0.0,
                }
                n_points = 0
                for batch in dataloader:
                    optimizer.zero_grad()
                    x, x_repr = batch
                    network_result = model(x)
                    assignments = network_result['assignments']
                    log_assignments = network_result['log_assignments']
                    old_assignments = model_copy(x)['assignments']

                    clust_loss = loss_func(x_repr, assignments, centroids=model.centroids)
                    kl_div = kld_loss(log_assignments, old_assignments) * self.kl_weight
                    loss = kl_div + clust_loss

                    loss.backward()
                    optimizer.step()
                    scheduler.step(loss)
                    epoch_loss['clust_loss'] += clust_loss.item()
                    epoch_loss['kl_div'] += kl_div.item()
                    n_points += x.shape[0]
                    p3_bar.set_postfix({'clust_loss': loss.item()})
                
                epoch_loss['clust_loss'] /= n_points
                epoch_loss['kl_div'] /= n_points
                epoch_losses.append(epoch_loss)
            model._lock_in_o1()

        if self.verbose:
            print('Starting Final Phase...')
            p4_bar = trange(self.epochs - self.warmup_epochs - self.o1_anneal_epochs - self.o2_anneal_epochs, desc='Training Phase')
        else:
            p4_bar = range(self.epochs - self.warmup_epochs - self.o1_anneal_epochs - self.o2_anneal_epochs)
        best_loss = np.inf
        best_ckpt = None
        best_epoch = 0
        for epoch in p4_bar:
            model.train()
            epoch_loss = {
                'epoch': epoch,
                'clust_loss': 0.0,
                'kl_div': 0.0,
            }
            n_points = 0
            for batch in dataloader:
                optimizer.zero_grad()
                x, x_repr = batch
                network_result = model(x)
                assignments = network_result['assignments']
                log_assignments = network_result['log_assignments']
                old_assignments = model_copy(x)['assignments']

                clust_loss = loss_func(x_repr, assignments, centroids=model.centroids)
                kl_div = kld_loss(log_assignments, old_assignments) * self.kl_weight
                loss = kl_div + clust_loss

                loss.backward()
                optimizer.step()
                scheduler.step(loss)
                epoch_loss['clust_loss'] += clust_loss.item()
                epoch_loss['kl_div'] += kl_div.item()
                n_points += x.shape[0]
                p4_bar.set_postfix({'clust_loss': loss.item()})
                
            epoch_loss['clust_loss'] /= n_points
            epoch_loss['kl_div'] /= n_points
            epoch_losses.append(epoch_loss)
            if epoch_loss['clust_loss'] < best_loss:
                best_loss = epoch_loss['clust_loss']
                best_ckpt = model.state_dict()
                best_epoch = epoch
            if epoch - best_epoch > 100:
                break
        model.load_state_dict(best_ckpt)
        self.model = model
        del model_copy
        gc.collect()
        torch.cuda.empty_cache()
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray):
        """
        Predict the soft cluster assignments for the input data. 
        Args:
            X (pd.DataFrame | np.ndarray): Input data in the interpretable space.
        Returns:
            pd.Series: Cluster assignments.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = torch.tensor(X, dtype=torch.float32)
        test_loader = DataLoader(X, batch_size=self.batch_size, shuffle=False)
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for x in test_loader:
                result = self.model(x)
                predictions.append(result['assignments'].cpu().numpy())
        predictions = np.concatenate(predictions, axis=0)
        return predictions
    def predict(self, X: pd.DataFrame | np.ndarray):
        """
        Predict the cluster assignments for the input data.
        Args:
            X (pd.DataFrame | np.ndarray): Input data in the interpretable space.
        Returns:
            pd.Series: Cluster assignments.
        """
        return np.argmax(self.predict_proba(X), axis=1)
    

class NeurCAMModel(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 repr_dim: int,
                 o1_channels: int,
                 o2_channels: int,
                 n_bases: int,
                 hidden_layers: list[int],
                 n_clusters):
        super(NeurCAMModel, self).__init__()
        self.input_dim = input_dim
        self.repr_dim = repr_dim
        self.o1_channels = o1_channels
        self.o2_channels = o2_channels
        self.n_bases = n_bases
        self.hidden_layers = hidden_layers
        self.n_clusters = n_clusters
        self.centroids = nn.Parameter(torch.zeros(n_clusters, self.repr_dim), requires_grad=True)
        nn.init.uniform_(self.centroids)

        if self.o1_channels > 0:
            self.o1_selection = nn.Parameter(torch.zeros(self.o1_channels, self.input_dim), requires_grad=True)
            self.o1_choice_temp = nn.Parameter(torch.tensor(1.0), requires_grad=False)
            nn.init.uniform_(self.o1_selection)
            if len(hidden_layers) == 0:
                layers = [nn.Linear(1, n_bases)]
            else:
                layers = [nn.Linear(1, hidden_layers[0])]
                for i in range(1, len(hidden_layers)):
                    layers.append(nn.ReLU())
                    layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
                
                layers.append(nn.ReLU())
                layers.append(nn.Linear(hidden_layers[-1], n_bases))
            self.o1_projection = nn.Sequential(*layers)
            self.o1_weights = nn.ModuleList([nn.Linear(n_bases, n_clusters) for _ in range(self.o1_channels)])


        if self.o2_channels > 0:
            self.o2_selection = nn.Parameter(torch.zeros(self.o2_channels, self.input_dim, 2), requires_grad=True)
            self.o2_choice_temp = nn.Parameter(torch.tensor(1.0), requires_grad=False)
            nn.init.uniform_(self.o2_selection)
            if len(hidden_layers) == 0:
                layers = [nn.Linear(2, n_bases)]
            else:
                layers = [nn.Linear(2, hidden_layers[0])]
                for i in range(1, len(hidden_layers)):
                    layers.append(nn.ReLU())
                    layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
                
                layers.append(nn.ReLU())
                layers.append(nn.Linear(hidden_layers[-1], n_bases))
            self.o2_projection = nn.Sequential(*layers)
            self.o2_weights = nn.ModuleList([nn.Linear(n_bases, n_clusters) for _ in range(self.o2_channels)])
            

        self.valid_cuts = False
        self.choice = Entmax15(dim=1)
        self.sm = nn.Softmax(dim=-1)
        self.log_sm = nn.LogSoftmax(dim=-1)

    def _initialize_centroids(self, train_loader, init_size = 10):
        """
        Similar to init_size argument in MB Kmeans 
        """
        # get first batch
        # x, x_repr = next(iter(train_loader))

        # get number of batches in train_loader
        n_batches = len(train_loader)
        init_size = min(init_size, n_batches)
        # get n_runs batches
        temp_centroids = torch.zeros_like(self.centroids)
        n_points = 0
        for i, (x, x_repr) in enumerate(train_loader):
            if i == init_size:
                break
            # get assignments
            W = self.forward(x)['assignments']
            # turn W into one-hot
            
            # use assignments to get the centroids
            centroids_num = torch.sum(W.unsqueeze(2) * x_repr.unsqueeze(1), axis=0)
            centroids_den = torch.sum(W, axis=0).unsqueeze(1)
            temp_centroids += centroids_num / centroids_den * x.shape[0]
            n_points += x.shape[0]
        temp_centroids /= n_points
        temp_centroids = temp_centroids.to(self.centroids.data.device)
       
        self.centroids.data = temp_centroids

    def _get_o1_selection(self):
        return self.choice(self.o1_selection / self.o1_choice_temp)
    
    def _get_o2_selection(self):
        return self.choice(self.o2_selection / self.o2_choice_temp)

    def _o1_valid_cuts(self):
        val_cuts_o1 = True
        total_nonzero = 0
        if self.o1_channels > 0:
            o1_selection = self._get_o1_selection()
            for i in range(self.o1_channels):
                n_non_zero = torch.count_nonzero(o1_selection[i,:])
                total_nonzero += n_non_zero
                val_cuts_o1 = val_cuts_o1 and n_non_zero <= 1
        return val_cuts_o1 

    def _o2_valid_cuts(self):
        val_cuts_o2 = True
        if self.o2_channels > 0:
            o2_selection = self._get_o2_selection()
            for i in range(self.o2_channels):
                rel_tensor1 = o2_selection[i,:,0]
                rel_tensor2 = o2_selection[i,:,1]
                n_non_zero1 = torch.count_nonzero(rel_tensor1)
                n_non_zero2 = torch.count_nonzero(rel_tensor2)
                val_cuts_o2 = val_cuts_o2 and n_non_zero1 <= 1 and n_non_zero2 <= 1
        return val_cuts_o2

    def _anneal_o1(self, o1_rel_epoch, o1_anneal_steps, min_temp):
        if self.o1_channels > 0:
            tau = min(o1_rel_epoch / o1_anneal_steps, 1.0)
            new_temperature = tau * np.log10(min_temp)
            self.o1_choice_temp.data = torch.tensor(10 ** new_temperature, dtype=torch.float32)

    def _anneal_o2(self, o2_rel_epoch, o2_anneal_steps, min_temp):
        if self.o2_channels > 0:
            tau = min(o2_rel_epoch / o2_anneal_steps, 1.0)
            new_temperature = tau * np.log10(min_temp)
            self.o2_choice_temp.data = torch.tensor(10 ** new_temperature, dtype=torch.float32)

    def _lock_in_o1(self):
        if self.o1_channels > 0:
            self.o1_selection.requires_grad = False
            self.o1_choice_temp.requires_grad = False
    def _lock_in_o2(self):
        if self.o2_channels > 0:
            self.o2_selection.requires_grad = False
            self.o2_choice_temp.requires_grad = False
    
    def forward(self, x):
        logits =  self._forward(x)
        assignments = self.sm(logits)
        log_assignments = self.log_sm(logits)
        return {
            'assignments': assignments,
            'log_assignments': log_assignments
        }

    def _seperated_forward_o1(self, X):
        o1_selection_weights = self._get_o1_selection()
        o1_select_save = F.linear(X, o1_selection_weights, bias=None)
        o1_select = o1_select_save.unsqueeze(2)
        # o1_select: (batch_size, o1_channels, 1)
        # o1_bases: (batch_size, o1_channels, n_bases)
        o1_bases = self.o1_projection(o1_select)
        results = {}
        for i in range(self.o1_channels):
            rel_selection = o1_selection_weights[i,:]
            # get the index of the non-zero element
            non_zero_index = torch.argmax(rel_selection).item()

            rel_bases = o1_bases[:,i,:]
            
            if non_zero_index not in results.keys():
                results[non_zero_index] = self.o1_weights[i](rel_bases)

            else:
                results[non_zero_index] += self.o1_weights[i](rel_bases)
        
        return results

    def _forward(self, X):
        # X: (batch_size, input_dim)
        # o1_selection: (input_dim, o1_channels, 1)
        result = torch.zeros(X.shape[0], self.n_clusters).to(X.device)
        if self.o1_channels> 0:
            o1_selection_weights = self._get_o1_selection()
            o1_select_save = F.linear(X, o1_selection_weights, bias=None)
            o1_select = o1_select_save.unsqueeze(2)
            # o1_select: (batch_size, o1_channels, 1)
            # o1_bases: (batch_size, o1_channels, n_bases)
            o1_bases = self.o1_projection(o1_select)

            for i in range(self.o1_channels):
                rel_bases = o1_bases[:,i,:]
                result += self.o1_weights[i](rel_bases)
        if self.o2_channels > 0:
            o2_selection_weights = self._get_o2_selection()
            o2_select = torch.einsum('bi,nio->bno', X, o2_selection_weights)
            o2_bases = self.o2_projection(o2_select)
            # o2_select: (batch_size, o2_channels, 2)
            # o2_bases: (batch_size, o2_channels, n_bases)
            for i in range(self.o2_channels):
                rel_bases = o2_bases[:,i,:]
                result += self.o2_weights[i](rel_bases)
        return result
