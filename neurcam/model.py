import gc
import random
from typing import Optional, Union, Dict, Any

import numpy as np
import pandas as pd
import torch
from entmax import Entmax15
from sklearn.cluster import KMeans, MiniBatchKMeans
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
import torch.nn.functional as F

from neurcam.loss import FuzzyCMeansLoss

# Constants
DEFAULT_RANDOM_STATE = 42
DEFAULT_M = 1.05
DEFAULT_HIDDEN_LAYERS = [128, 128]
DEFAULT_N_BASES = 64
DEFAULT_LEARNING_RATE = 2e-3
DEFAULT_EPOCHS = 5000
DEFAULT_BATCH_SIZE = 512
DEFAULT_WARMUP_RATIO = 0.4
DEFAULT_O1_ANNEAL_RATIO = 0.1
DEFAULT_O2_ANNEAL_RATIO = 0.1
DEFAULT_MIN_TEMP = 1e-5
DEFAULT_KL_WEIGHT = 1.0
DEFAULT_MODEL_DIR = "NeurCAMCheckpoints"
DEFAULT_PATIENCE = 100
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 50
CENTROID_INIT_SIZE = 3


class NeurCAM:
    def __init__(
        self,
        k: int,
        random_state: int = DEFAULT_RANDOM_STATE,
        m: float = DEFAULT_M,
        hidden_layers: list[int] = None,
        n_bases: int = DEFAULT_N_BASES,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        single_feature_channels: Union[float, int] = 1.0,
        pairwise_feature_channels: Union[float, int] = 0.0,
        warmup_ratio: Union[float, int] = DEFAULT_WARMUP_RATIO,
        o1_anneal_ratio: Union[float, int] = DEFAULT_O1_ANNEAL_RATIO,
        o2_anneal_ratio: Union[float, int] = DEFAULT_O2_ANNEAL_RATIO,
        min_temp: float = DEFAULT_MIN_TEMP,
        kl_weight: float = DEFAULT_KL_WEIGHT,
        smart_init: str = "none",
        model_dir: str = DEFAULT_MODEL_DIR,
        device: str = "auto",
        verbose: bool = True,
    ) -> None:
        """
        NeurCAM class for interpretable clustering.

        Args:
            k: Number of clusters.
            random_state: Random seed for reproducibility.
            m: Fuzziness parameter.
            hidden_layers: List of hidden layer dimensions for the backbone network.
            n_bases: Output dimension of the backbone.
            learning_rate: Learning rate for the optimizer.
            epochs: Total number of training epochs.
            batch_size: Batch size for training.
            single_feature_channels: Number of channels for single feature interactions.
                If values are <=1.0, interpreted as ratio of number of features.
                If values are >1.0, interpreted as number of channels.
            pairwise_feature_channels: Number of channels for pairwise feature interactions.
            warmup_ratio: Ratio of warmup epochs.
            o1_anneal_ratio: Ratio of first annealing phase.
            o2_anneal_ratio: Ratio of second annealing phase.
            min_temp: Minimum temperature for annealing.
            kl_weight: Weight for the KL divergence loss.
            smart_init: Clustering initialization method ('none', 'kmeans', 'mbkmeans').
            model_dir: Directory to save model checkpoints.
            device: Device to use for training ('auto', 'cuda', 'cpu').
            verbose: Whether to print training progress.
        """
        if hidden_layers is None:
            hidden_layers = DEFAULT_HIDDEN_LAYERS.copy()

        # Validate parameters
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if m < 1.0:
            raise ValueError(f"m must be >= 1.0 for fuzzy clustering, got {m}")
        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if n_bases <= 0:
            raise ValueError(f"n_bases must be positive, got {n_bases}")
        if min_temp <= 0:
            raise ValueError(f"min_temp must be positive, got {min_temp}")
        if kl_weight < 0:
            raise ValueError(f"kl_weight must be non-negative, got {kl_weight}")
        if single_feature_channels < 0:
            raise ValueError(
                f"single_feature_channels must be non-negative, got {single_feature_channels}"
            )
        if pairwise_feature_channels < 0:
            raise ValueError(
                f"pairwise_feature_channels must be non-negative, got {pairwise_feature_channels}"
            )
        if not (0 <= warmup_ratio <= 1):
            raise ValueError(f"warmup_ratio must be in [0, 1], got {warmup_ratio}")
        if not (0 <= o1_anneal_ratio <= 1):
            raise ValueError(f"o1_anneal_ratio must be in [0, 1], got {o1_anneal_ratio}")
        if not (0 <= o2_anneal_ratio <= 1):
            raise ValueError(f"o2_anneal_ratio must be in [0, 1], got {o2_anneal_ratio}")
        if smart_init not in ["none", "kmeans", "mbkmeans"]:
            raise ValueError(
                f"smart_init must be one of ['none', 'kmeans', 'mbkmeans'], got '{smart_init}'"
            )
        # Validate device is a valid PyTorch device specifier
        # Allow "auto", or any string that torch.device() accepts
        if device != "auto":
            try:
                torch.device(device)
            except RuntimeError as e:
                raise ValueError(f"Invalid device specifier: '{device}'. Error: {e}")

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
        self.model: Optional[NeurCAMModel] = None
        self.verbose = verbose

        self.warmup_epochs = int(self.epochs * self.warmup_ratio)
        self.o1_anneal_epochs = int(self.epochs * self.o1_anneal_ratio)
        self.o2_anneal_epochs = int(self.epochs * self.o2_anneal_ratio)
        self.feature_names: Optional[pd.Index] = None
        self.n_features = -1
        self.repr_dim = -1
        self.device = device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _set_random_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        random.seed(self.random_state)

    def _prepare_data(
        self, X: Union[pd.DataFrame, np.ndarray], X_repr: Optional[Union[pd.DataFrame, np.ndarray]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare input data for training.

        Args:
            X: Input data in the interpretable space.
            X_repr: Input data in transformed/latent space (optional).

        Returns:
            Tuple of (X, X_repr) as numpy arrays.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns
            X = X.values
        if X_repr is not None:
            if isinstance(X_repr, pd.DataFrame):
                X_repr = X_repr.values
            if X.shape[0] != X_repr.shape[0]:
                raise ValueError(
                    f"X and X_repr must have the same number of samples. "
                    f"Got X.shape[0]={X.shape[0]} and X_repr.shape[0]={X_repr.shape[0]}"
                )
        else:
            X_repr = X

        if X.shape[0] < self.k:
            raise ValueError(
                f"Number of samples ({X.shape[0]}) must be at least as large as "
                f"number of clusters ({self.k})"
            )

        return X, X_repr

    def _compute_channel_counts(self) -> None:
        """Compute the actual number of channels based on ratios or absolute values."""
        if self.sf_channels <= 1.0:
            self.single_feature_channels = max(0, int(self.sf_channels * self.n_features))
        else:
            self.single_feature_channels = int(self.sf_channels)

        if self.pf_channels <= 1.0:
            self.pairwise_feature_channels = max(0, int(self.pf_channels * self.n_features))
        else:
            self.pairwise_feature_channels = int(self.pf_channels)

    def _create_model(self) -> "NeurCAMModel":
        """Create and initialize the NeurCAM model."""
        model = NeurCAMModel(
            input_dim=self.n_features,
            repr_dim=self.repr_dim,
            o1_channels=self.single_feature_channels,
            o2_channels=self.pairwise_feature_channels,
            n_bases=self.n_bases,
            hidden_layers=self.hidden_layers,
            n_clusters=self.k,
        )
        return model.to(self.device)

    def _initialize_centroids(
        self, model: "NeurCAMModel", X_repr: np.ndarray, dataloader: DataLoader
    ) -> None:
        """
        Initialize cluster centroids using the specified method.

        Args:
            model: The NeurCAM model to initialize.
            X_repr: Input data in representation space.
            dataloader: DataLoader for the training data.
        """
        if self.smart_init == "kmeans":
            kmeans = KMeans(n_clusters=self.k, random_state=self.random_state, n_init=10)
            kmeans.fit(X_repr)
            tens = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
            model.centroids.data = tens.to(model.centroids.data.device)

        elif self.smart_init == "mbkmeans":
            kmeans = MiniBatchKMeans(n_clusters=self.k, random_state=self.random_state, n_init=10)
            kmeans.fit(X_repr)
            tens = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
            model.centroids.data = tens.to(model.centroids.data.device)
        else:
            model._initialize_centroids(dataloader, init_size=CENTROID_INIT_SIZE)

    def _train_epoch(
        self,
        model: "NeurCAMModel",
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: "torch.optim.lr_scheduler.LRScheduler",
        loss_func: FuzzyCMeansLoss,
        kld_loss: nn.KLDivLoss,
        model_copy: Optional["NeurCAMModel"] = None,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            model: The model to train.
            dataloader: DataLoader for training data.
            optimizer: Optimizer for model parameters.
            scheduler: Learning rate scheduler.
            loss_func: Fuzzy C-means loss function.
            kld_loss: KL divergence loss function.
            model_copy: Optional copy of model for KL divergence calculation.

        Returns:
            Dictionary containing loss values for the epoch.
        """
        model.train()
        epoch_loss = {"clust_loss": 0.0, "kl_div": 0.0}
        n_points = 0

        for batch in dataloader:
            optimizer.zero_grad()
            x, x_repr = batch
            network_result = model(x)
            assignments = network_result["assignments"]

            clust_loss = loss_func(x_repr, assignments, centroids=model.centroids)
            loss = clust_loss

            # Add KL divergence if model_copy is provided
            if model_copy is not None:
                log_assignments = network_result["log_assignments"]
                old_assignments = model_copy(x)["assignments"]
                kl_div = kld_loss(log_assignments, old_assignments) * self.kl_weight
                loss = loss + kl_div
                epoch_loss["kl_div"] += kl_div.item()

            loss.backward()
            optimizer.step()

            epoch_loss["clust_loss"] += clust_loss.item()
            n_points += x.shape[0]

        epoch_loss["clust_loss"] /= n_points
        if model_copy is not None:
            epoch_loss["kl_div"] /= n_points

        # Call scheduler once per epoch with the epoch-averaged loss
        scheduler.step(epoch_loss["clust_loss"])

        return epoch_loss

    def _train_phase(
        self,
        model: "NeurCAMModel",
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: "torch.optim.lr_scheduler.LRScheduler",
        loss_func: FuzzyCMeansLoss,
        kld_loss: nn.KLDivLoss,
        n_epochs: int,
        phase_name: str,
        model_copy: Optional["NeurCAMModel"] = None,
        track_best: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Run a complete training phase.

        Args:
            model: The model to train.
            dataloader: DataLoader for training data.
            optimizer: Optimizer for model parameters.
            scheduler: Learning rate scheduler.
            loss_func: Fuzzy C-means loss function.
            kld_loss: KL divergence loss function.
            n_epochs: Number of epochs for this phase.
            phase_name: Name of the training phase for logging.
            model_copy: Optional copy of model for KL divergence calculation.
            track_best: Whether to track and return the best checkpoint.

        Returns:
            Best model state dict if track_best is True, None otherwise.
        """
        if self.verbose:
            print(f"Starting {phase_name}...")
            progress_bar = trange(n_epochs, desc=phase_name)
        else:
            progress_bar = range(n_epochs)

        best_loss = np.inf
        best_ckpt = None
        best_epoch = 0

        for epoch in progress_bar:
            epoch_loss = self._train_epoch(
                model, dataloader, optimizer, scheduler, loss_func, kld_loss, model_copy
            )

            if self.verbose:
                progress_bar.set_postfix({"clust_loss": epoch_loss["clust_loss"]})

            if track_best and epoch_loss["clust_loss"] < best_loss:
                best_loss = epoch_loss["clust_loss"]
                best_ckpt = model.state_dict()
                best_epoch = epoch

            if track_best and epoch - best_epoch > DEFAULT_PATIENCE:
                break

        return best_ckpt if track_best else None

    def _train_annealing_phase(
        self,
        model: "NeurCAMModel",
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: "torch.optim.lr_scheduler.LRScheduler",
        loss_func: FuzzyCMeansLoss,
        kld_loss: nn.KLDivLoss,
        model_copy: "NeurCAMModel",
        n_epochs: int,
        phase_name: str,
        anneal_fn: str,
        valid_cuts_fn: str,
        lock_fn: str,
    ) -> None:
        """
        Run an annealing training phase.

        Args:
            model: The model to train.
            dataloader: DataLoader for training data.
            optimizer: Optimizer for model parameters.
            scheduler: Learning rate scheduler.
            loss_func: Fuzzy C-means loss function.
            kld_loss: KL divergence loss function.
            model_copy: Copy of model for KL divergence calculation.
            n_epochs: Number of epochs for this phase.
            phase_name: Name of the training phase for logging.
            anneal_fn: Name of annealing method to call.
            valid_cuts_fn: Name of validation method to check.
            lock_fn: Name of lock-in method to call after training.
        """
        if self.verbose:
            print(f"Starting {phase_name}...")
            progress_bar = trange(n_epochs, desc=phase_name)
        else:
            progress_bar = range(n_epochs)

        for epoch in progress_bar:
            model.train()
            getattr(model, anneal_fn)(epoch, n_epochs, self.min_temp)

            valid_cuts = getattr(model, valid_cuts_fn)()
            if valid_cuts:
                break

            epoch_loss = self._train_epoch(
                model, dataloader, optimizer, scheduler, loss_func, kld_loss, model_copy
            )

            if self.verbose:
                progress_bar.set_postfix({"clust_loss": epoch_loss["clust_loss"]})

        getattr(model, lock_fn)()

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        X_repr: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    ) -> "NeurCAM":
        """
        Fit the NeurCAM model.

        Args:
            X: Input data in the interpretable space.
            X_repr: Input data in transformed/latent space (optional).

        Returns:
            Self for method chaining.
        """
        # Set random seeds for reproducibility
        self._set_random_seeds()

        # Prepare data
        X, X_repr = self._prepare_data(X, X_repr)
        self.n_features = X.shape[1]
        self.repr_dim = X_repr.shape[1]

        # Create dataset and dataloader
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32, device=self.device),
            torch.tensor(X_repr, dtype=torch.float32, device=self.device),
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Compute channel counts
        self._compute_channel_counts()

        # Create and initialize model
        model = self._create_model()
        self._initialize_centroids(model, X_repr, dataloader)

        # Setup training components
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE
        )
        loss_func = FuzzyCMeansLoss(m=self.m)
        kld_loss = nn.KLDivLoss(reduction="batchmean")

        # Warmup phase
        best_ckpt = self._train_phase(
            model,
            dataloader,
            optimizer,
            scheduler,
            loss_func,
            kld_loss,
            self.warmup_epochs,
            "Warmup Phase",
            model_copy=None,
            track_best=True,
        )

        # Load best checkpoint and create copy for KL divergence
        model.load_state_dict(best_ckpt)
        model_copy = self._create_model()
        model_copy.load_state_dict(best_ckpt)
        model_copy.eval()
        del best_ckpt
        gc.collect()

        # Pairwise feature annealing phase
        if self.pairwise_feature_channels > 0:
            self._train_annealing_phase(
                model,
                dataloader,
                optimizer,
                scheduler,
                loss_func,
                kld_loss,
                model_copy,
                self.o2_anneal_epochs,
                "O2 Annealing Phase",
                "_anneal_o2",
                "_o2_valid_cuts",
                "_lock_in_o2",
            )

        # Single feature annealing phase
        if self.single_feature_channels > 0:
            self._train_annealing_phase(
                model,
                dataloader,
                optimizer,
                scheduler,
                loss_func,
                kld_loss,
                model_copy,
                self.o1_anneal_epochs,
                "O1 Annealing Phase",
                "_anneal_o1",
                "_o1_valid_cuts",
                "_lock_in_o1",
            )

        # Final training phase
        final_epochs = (
            self.epochs - self.warmup_epochs - self.o1_anneal_epochs - self.o2_anneal_epochs
        )
        best_ckpt = self._train_phase(
            model,
            dataloader,
            optimizer,
            scheduler,
            loss_func,
            kld_loss,
            final_epochs,
            "Final Training Phase",
            model_copy=model_copy,
            track_best=True,
        )

        # Load best checkpoint
        model.load_state_dict(best_ckpt)
        self.model = model

        # Cleanup
        del model_copy
        gc.collect()
        torch.cuda.empty_cache()

        return self

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict the soft cluster assignments for the input data.

        Args:
            X: Input data in the interpretable space.

        Returns:
            Numpy array of soft cluster assignments with shape (n_samples, n_clusters).
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        test_loader = DataLoader(X, batch_size=self.batch_size, shuffle=False)
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for x in test_loader:
                result = self.model(x)
                predictions.append(result["assignments"].cpu().numpy())
        predictions = np.concatenate(predictions, axis=0)
        return predictions

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict the cluster assignments for the input data.

        Args:
            X: Input data in the interpretable space.

        Returns:
            Numpy array of hard cluster assignments with shape (n_samples,).
        """
        return np.argmax(self.predict_proba(X), axis=1)


class NeurCAMModel(nn.Module):
    """
    Neural network model for NeurCAM clustering.

    This model implements a neural clustering approach with support for
    single-feature and pairwise-feature interactions.
    """

    def __init__(
        self,
        input_dim: int,
        repr_dim: int,
        o1_channels: int,
        o2_channels: int,
        n_bases: int,
        hidden_layers: list[int],
        n_clusters: int,
    ) -> None:
        """
        Initialize the NeurCAM model.

        Args:
            input_dim: Dimension of input features.
            repr_dim: Dimension of representation space.
            o1_channels: Number of single-feature channels.
            o2_channels: Number of pairwise-feature channels.
            n_bases: Number of basis functions.
            hidden_layers: List of hidden layer dimensions.
            n_clusters: Number of clusters.
        """
        super(NeurCAMModel, self).__init__()

        # Validate parameters
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if repr_dim <= 0:
            raise ValueError(f"repr_dim must be positive, got {repr_dim}")
        if o1_channels < 0:
            raise ValueError(f"o1_channels must be non-negative, got {o1_channels}")
        if o2_channels < 0:
            raise ValueError(f"o2_channels must be non-negative, got {o2_channels}")
        if n_bases <= 0:
            raise ValueError(f"n_bases must be positive, got {n_bases}")
        if n_clusters <= 0:
            raise ValueError(f"n_clusters must be positive, got {n_clusters}")

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
            self._initialize_o1_layers()

        if self.o2_channels > 0:
            self._initialize_o2_layers()

        self.valid_cuts = False
        self.choice = Entmax15(dim=1)
        self.sm = nn.Softmax(dim=-1)
        self.log_sm = nn.LogSoftmax(dim=-1)

    def _build_projection_network(self, input_size: int) -> nn.Sequential:
        """
        Build a projection network with the specified architecture.

        Args:
            input_size: Size of the input layer.

        Returns:
            Sequential neural network module.
        """
        layers = []

        if not self.hidden_layers:
            layers.append(nn.Linear(input_size, self.n_bases))
        else:
            layers.append(nn.Linear(input_size, self.hidden_layers[0]))

            for i in range(1, len(self.hidden_layers)):
                layers.extend(
                    [nn.ReLU(), nn.Linear(self.hidden_layers[i - 1], self.hidden_layers[i])]
                )

            layers.extend([nn.ReLU(), nn.Linear(self.hidden_layers[-1], self.n_bases)])

        return nn.Sequential(*layers)

    def _initialize_o1_layers(self) -> None:
        """Initialize layers for single-feature processing."""
        self.o1_selection = nn.Parameter(
            torch.zeros(self.o1_channels, self.input_dim), requires_grad=True
        )
        self.o1_choice_temp = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        nn.init.uniform_(self.o1_selection)
        self.o1_projection = self._build_projection_network(input_size=1)
        self.o1_weights = nn.ModuleList(
            [nn.Linear(self.n_bases, self.n_clusters) for _ in range(self.o1_channels)]
        )

    def _initialize_o2_layers(self) -> None:
        """Initialize layers for pairwise-feature processing."""
        self.o2_selection = nn.Parameter(
            torch.zeros(self.o2_channels, self.input_dim, 2), requires_grad=True
        )
        self.o2_choice_temp = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        nn.init.uniform_(self.o2_selection)
        self.o2_projection = self._build_projection_network(input_size=2)
        self.o2_weights = nn.ModuleList(
            [nn.Linear(self.n_bases, self.n_clusters) for _ in range(self.o2_channels)]
        )

    def _initialize_centroids(self, train_loader: DataLoader, init_size: int = 10) -> None:
        """
        Initialize centroids using initial batches from the training data.

        Similar to init_size argument in MiniBatch KMeans.

        Args:
            train_loader: DataLoader for training data.
            init_size: Number of batches to use for initialization.
        """
        n_batches = len(train_loader)
        init_size = min(init_size, n_batches)
        temp_centroids = torch.zeros_like(self.centroids)
        n_points = 0

        for i, (x, x_repr) in enumerate(train_loader):
            if i == init_size:
                break
            W = self.forward(x)["assignments"]
            centroids_num = torch.sum(W.unsqueeze(2) * x_repr.unsqueeze(1), axis=0)
            centroids_den = torch.sum(W, axis=0).unsqueeze(1)
            temp_centroids += centroids_num / centroids_den * x.shape[0]
            n_points += x.shape[0]

        temp_centroids /= n_points
        temp_centroids = temp_centroids.to(self.centroids.data.device)
        self.centroids.data = temp_centroids

    def _get_o1_selection(self) -> torch.Tensor:
        """Get single-feature selection weights with temperature scaling."""
        return self.choice(self.o1_selection / self.o1_choice_temp)

    def _get_o2_selection(self) -> torch.Tensor:
        """Get pairwise-feature selection weights with temperature scaling."""
        return self.choice(self.o2_selection / self.o2_choice_temp)

    def _o1_valid_cuts(self) -> bool:
        """
        Check if single-feature selections are valid (at most one feature per channel).

        Returns:
            True if all channels have at most one selected feature.
        """
        if self.o1_channels == 0:
            return True

        o1_selection = self._get_o1_selection()
        non_zero_counts = torch.count_nonzero(o1_selection, dim=1)
        return torch.all(non_zero_counts <= 1).item()

    def _o2_valid_cuts(self) -> bool:
        """
        Check if pairwise-feature selections are valid (at most one feature per slot).

        Returns:
            True if all channels have at most one selected feature per slot.
        """
        if self.o2_channels == 0:
            return True

        o2_selection = self._get_o2_selection()
        non_zero_counts_slot0 = torch.count_nonzero(o2_selection[:, :, 0], dim=1)
        non_zero_counts_slot1 = torch.count_nonzero(o2_selection[:, :, 1], dim=1)
        return (
            torch.all(non_zero_counts_slot0 <= 1) and torch.all(non_zero_counts_slot1 <= 1)
        ).item()

    def _anneal_o1(self, o1_rel_epoch: int, o1_anneal_steps: int, min_temp: float) -> None:
        """
        Anneal the temperature for single-feature selection.

        Args:
            o1_rel_epoch: Current epoch in the annealing phase.
            o1_anneal_steps: Total number of annealing steps.
            min_temp: Minimum temperature to reach.
        """
        if self.o1_channels > 0:
            tau = min(o1_rel_epoch / o1_anneal_steps, 1.0)
            new_temperature = tau * np.log10(min_temp)
            self.o1_choice_temp.data = torch.tensor(10**new_temperature, dtype=torch.float32)

    def _anneal_o2(self, o2_rel_epoch: int, o2_anneal_steps: int, min_temp: float) -> None:
        """
        Anneal the temperature for pairwise-feature selection.

        Args:
            o2_rel_epoch: Current epoch in the annealing phase.
            o2_anneal_steps: Total number of annealing steps.
            min_temp: Minimum temperature to reach.
        """
        if self.o2_channels > 0:
            tau = min(o2_rel_epoch / o2_anneal_steps, 1.0)
            new_temperature = tau * np.log10(min_temp)
            self.o2_choice_temp.data = torch.tensor(10**new_temperature, dtype=torch.float32)

    def _lock_in_o1(self) -> None:
        """Lock single-feature selections by disabling gradient updates."""
        if self.o1_channels > 0:
            self.o1_selection.requires_grad = False
            self.o1_choice_temp.requires_grad = False

    def _lock_in_o2(self) -> None:
        """Lock pairwise-feature selections by disabling gradient updates."""
        if self.o2_channels > 0:
            self.o2_selection.requires_grad = False
            self.o2_choice_temp.requires_grad = False

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Dictionary containing 'assignments' and 'log_assignments'.
        """
        logits = self._forward(x)
        assignments = self.sm(logits)
        log_assignments = self.log_sm(logits)
        return {"assignments": assignments, "log_assignments": log_assignments}

    def _separated_forward_o1(self, X: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Compute single-feature contributions separately for each feature.

        Args:
            X: Input tensor of shape (batch_size, input_dim).

        Returns:
            Dictionary mapping feature indices to their contributions.
        """
        o1_selection_weights = self._get_o1_selection()
        o1_select_save = F.linear(X, o1_selection_weights, bias=None)
        o1_select = o1_select_save.unsqueeze(2)
        o1_bases = self.o1_projection(o1_select)
        results = {}

        for i in range(self.o1_channels):
            rel_selection = o1_selection_weights[i, :]
            non_zero_index = torch.argmax(rel_selection).item()
            rel_bases = o1_bases[:, i, :]

            if non_zero_index in results:
                results[non_zero_index] += self.o1_weights[i](rel_bases)
            else:
                results[non_zero_index] = self.o1_weights[i](rel_bases)

        return results

    def _forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Internal forward pass computing logits.

        Args:
            X: Input tensor of shape (batch_size, input_dim).

        Returns:
            Logits tensor of shape (batch_size, n_clusters).
        """
        result = torch.zeros(X.shape[0], self.n_clusters, device=X.device)

        if self.o1_channels > 0:
            o1_selection_weights = self._get_o1_selection()
            o1_select_save = F.linear(X, o1_selection_weights, bias=None)
            o1_select = o1_select_save.unsqueeze(2)
            o1_bases = self.o1_projection(o1_select)

            for i in range(self.o1_channels):
                rel_bases = o1_bases[:, i, :]
                result += self.o1_weights[i](rel_bases)

        if self.o2_channels > 0:
            o2_selection_weights = self._get_o2_selection()
            o2_select = torch.einsum("bi,nio->bno", X, o2_selection_weights)
            o2_bases = self.o2_projection(o2_select)

            for i in range(self.o2_channels):
                rel_bases = o2_bases[:, i, :]
                result += self.o2_weights[i](rel_bases)

        return result
