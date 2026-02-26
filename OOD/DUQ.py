import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau
import optuna
from sklearn.metrics import roc_auc_score
import json
from torch.utils.data import DataLoader
import numpy as np


class DUQModel(nn.Module):
    """
    Deterministic Uncertainty Quantification (DUQ) model head.

    Fits a per-class RBF kernel on top of a feature extractor, as proposed in:
    https://proceedings.mlr.press/v119/van-amersfoort20a.html

    Features:
        - Uses moving averages to track class centroids in the feature space.
        - Learnable sigma parameter for the RBF kernel (enforced positive via
            softplus).
        - Weight initialization scaled for ELU activations.

    References:
        - Implementation adapted from:
          https://github.com/theresabruns/UncertaintyEstimation/blob/main/DUQ/utils/resnet_duq.py
          https://github.com/y0ast/deterministic-uncertainty-quantification
        - ELU initialization discussion:
          https://stats.stackexchange.com/questions/229885/whats-the-recommended-weight-initialization-strategy-when-using-the-elu-activat?
    """
    def __init__(self,
                 feature_extractor: nn.Module,
                 num_classes: int,
                 centroid_size: int = 32,
                 sigma: float = 0.1,
                 gamma: float = 0.999
                 ) -> None:
        """
        Initialize the DUQ model.

        Args:
            feature_extractor (nn.Module): Pretrained feature extractor.
                The flattened features should come from the penultimate layer.
            num_classes (int): Number of in-distribution (ID) classes.
            centroid_size (int, optional): Dimensionality of per-class
                centroids. Defaults to 32.
            sigma (float, optional): Initial value of log_sigma parameter
                shaping the RBF kernel. Defaults to 0.1.
            gamma (float, optional): Moving average update rate for centroids.
                New information is added at a rate of 1 - gamma.
                Defaults to 0.999.
        """

        super().__init__()

        self.feature_extractor = feature_extractor

        self.thresholds = None

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.gamma = gamma
        self.log_sigma = nn.Parameter(
                torch.log(torch.ones(num_classes) * sigma)
            )

        linear_size = feature_extractor.linear_size
        self.W = nn.Parameter(
            torch.zeros(
                centroid_size, num_classes, linear_size
                )
            )

        gain_elu = math.sqrt(1.55)
        fan_in = nn.init._calculate_correct_fan(self.W, mode="fan_in")
        std = gain_elu / math.sqrt(fan_in)
        with torch.no_grad():
            self.W.normal_(0, std=std)

        self.register_buffer("moving_count", torch.ones(num_classes)*12)
        self.register_buffer("moving_average", torch.normal(
            torch.zeros(centroid_size, num_classes), 0.05
            )
        )
        self.moving_average.mul_(self.moving_count.unsqueeze(0))

    def _rbf(self,
             z: torch.Tensor
             ) -> torch.Tensor:
        """
        The rbf kernel calculating the similarity score
        of a sample to the learned ID class distributions.

        Args:
            z (torch.Tensor): A feature vector.

        Returns:
            torch.Tensor: The similarity of the provided feature vector
                to the learned class distributions.
        """

        z = torch.einsum("ij,mnj->imn", z, self.W)

        sigma = F.softplus(self.log_sigma) + 1e-6
        sigma = sigma.view(1, -1)

        embeddings = self.moving_average / self.moving_count.unsqueeze(0)

        similarity = z - embeddings.unsqueeze(0)
        dist = similarity.pow(2).mean(dim=1)
        similarity_score = torch.exp(-dist / (2 * sigma**2))

        return similarity_score

    def update_embeddings(self,
                          input: torch.Tensor,
                          label: torch.Tensor
                          ) -> None:
        """
        Updates the per class moving averages.

        Args:
            input (torch.Tensor): Feature vectors from the train set.
            label (torch.Tensor): The labels belonging to the provided
                feature vectors.
        """

        z = input

        self.moving_count.mul_(self.gamma).add_(
            (1 - self.gamma) * label.sum(0)
        )

        z = torch.einsum("ij,mnj->imn", z, self.W)
        embedding_sum = torch.einsum("ijk,ik->jk", z, label)

        self.moving_average.mul_(self.gamma).add_(
            (1 - self.gamma) * embedding_sum
        )

    def feed_features(self,
                      features: torch.Tensor
                      ) -> torch.Tensor:
        """
        A forward pass through the DUQ model head
        from provided feature vectors.

        Args:
            features (torch.Tensor): Feature vectors.

        Returns:
            torch.Tensor: The similarity of the provided feature vector
                to the learned class distributions.
        """

        prediction = self._rbf(features)
        return prediction

    def forward(self,
                input: torch.Tensor
                ) -> torch.Tensor:
        """
        A forward pass through both the feature extractor and the
        DUQ model head.

        Args:
            input (torch.Tensor): A raw input from which the
                features still need to be extracted.

        Returns:
            torch.Tensor: The similarity of the provided feature vector
                to the learned class distributions.
        """

        with torch.no_grad():
            features = self.feature_extractor.feature_extractor(input)

        if self.thresholds is not None:
            features = torch.minimum(features, self.thresholds)

        return self.feed_features(features)


class TrainDUQ():
    """
    Trainer class for the DUQ model head.

    Handles training of the DUQ head on top of a frozen feature extractor.
    Uses a ReduceLROnPlateau scheduler to adjust the learning rate based on
    validation loss with a patience of 5 epochs and a factor of 0.5.

    Attributes:
        model (nn.Module): DUQ model including frozen feature extractor.
        optimizer (Optimizer): Optimizer used to train the DUQ head.
        loss_criterion (Callable): Loss function to optimize.
        device (torch.device): Device to run training on.
        n_classes (int): Number of in-distribution (ID) classes.
        scheduler (ReduceLROnPlateau): Learning rate scheduler for training.
    """
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 n_classes: int = 3,
                 loss_criterion:
                 torch.nn.modules.loss._Loss = F.binary_cross_entropy,
                 device: torch.cuda.device = torch.device(
                     'cuda' if torch.cuda.is_available() else 'cpu')
                 ) -> None:
        """
        Initialize the DUQ trainer.

        Args:
            model (nn.Module): Trained feature extractor model with a DUQ head.
                Flattened features should come from the penultimate layer.
            optimizer (Optimizer): Optimizer for training the DUQ head.
            n_classes (int, optional): Number of in-distribution (ID) classes.
                Defaults to 3.
            loss_criterion (nn.modules.loss._Loss, optional): Loss function to
                optimize. Defaults to F.binary_cross_entropy.
            device (torch.device, optional): Device for training
                ('cuda' or 'cpu'). Defaults to CUDA if available.
        """

        self.model = model.to(device)
        self.loss_criterion = loss_criterion
        self.optimizer = optimizer
        self.device = device
        self.n_classes = n_classes
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    def _compute_gradient_penalty(self,
                                  inputs: torch.Tensor,
                                  outputs: torch.Tensor
                                  ) -> torch.Tensor:
        """
        Compute the two-sided gradient penalty to stabilize training.

        Args:
            inputs (torch.Tensor): The feature vector serving as input
                to the DUQ model head.
            outputs (torch.Tensor): The outputs of the DUQ model head.

        Returns:
            torch.Tensor: The gradient penalty.
        """
        gradients = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
        )[0]

        gradients = gradients.flatten(start_dim=1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()
        return penalty

    def test(self,
             test_loader: DataLoader
             ) -> float:
        """
        Evaluate the DUQ model on the test set.

        Args:
            test_loader (DataLoader): A PyTorch Dataloader
                object storing data from the validation split.

        Returns:
            float: The average loss on the validation split
                used for early stopping.
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                oh_labels = F.one_hot(labels, self.n_classes).float()
                outputs = self.model(inputs)
                loss = self.loss_criterion(outputs, oh_labels, reduction="sum")

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                total_correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(test_loader)
        return avg_loss

    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              n_epochs: int = 100,
              penalty_w: float = 0.05,
              patience: int = 10,
              threshold: None | torch.Tensor = None,
              trial: None | int = None
              ) -> nn.Module:
        """
        A training loop for the DUQ model using a two-sided gradient
        penalty and early stopping.

        Args:
            train_loader (DataLoader): A PyTorch Dataloader
                object storing data from the train split.
            val_loader (DataLoader): A PyTorch Dataloader
                object storing data from the validation split.
            n_epochs (int, optional): Number of epochs for which to train the
                model head. Defaults to 100.
            penalty_w (float, optional): The magnitude of the gradient
                penalty. Defaults to 0.05.
            patience (int, optional): Defines how long to resume model
                training when the criterion on the validation set does not
                improve. Defaults to 10.
            threshold (None | torch.Tensor, optional):  The thresholds defined
                for the feature extractor when clipping activations using
                ReAct. Defaults to None.
            trial (None | int, optional): When tuning the hyperparameters
                using Optune this defines which trail it is at.
                Defaults to None.

        Raises:
            optuna.TrialPruned: Prunes model training if the loss is not
                decreasing fast enough as seen in previous trials.

        Returns:
            nn.Module: The DUQ head found during training that performed
                best on the ID validation set.
        """

        best_val_loss = float('inf')
        no_improve = 0

        self.model.thresholds = threshold
        if threshold is not None:
            print("ReAct activated in DUQ during training")

        for epoch in range(n_epochs):
            total_loss = 0.0

            for _, (inputs, labels) in enumerate(train_loader):

                self.model.train()

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                oh_labels = F.one_hot(labels, self.n_classes).float()

                with torch.no_grad():

                    features = self.model.feature_extractor.feature_extractor(
                        inputs)
                    if self.model.thresholds is not None:
                        features = torch.minimum(
                            features, self.model.thresholds)

                features = features.detach()
                features.requires_grad_(True)

                outputs = self.model.feed_features(features)
                loss = self.loss_criterion(outputs, oh_labels, reduction="sum")

                output_sum = outputs.sum(dim=1)
                gradient_penalty = self._compute_gradient_penalty(
                    features, output_sum)
                loss += penalty_w * gradient_penalty

                total_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                features.requires_grad_(False)
                with torch.no_grad():
                    self.model.eval()
                    self.model.update_embeddings(features, oh_labels)

            val_loss = self.test(val_loader)

            if trial is not None:
                trial.report(-val_loss, epoch)

                if trial.should_prune():
                    print("Pruned at epoch", epoch)
                    raise optuna.TrialPruned()

            self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                duq_state = {
                    k: v for k, v in self.model.state_dict().items()
                    if not k.startswith("feature_extractor")
                }
                best_model_weights = duq_state
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print('Early stopping at epoch {} \
                          when training DUQ'.format(epoch))
                    break

        if best_model_weights:
            self.model.load_state_dict(best_model_weights, strict=False)
            print('Loaded best model weights.')

        self.model.eval()
        return self.model


class DUQ():
    """
    Wrapper class for Deterministic Uncertainty Quantification (DUQ).

    This class stores:
        - The DUQ model head (DUQModel) built on top of a frozen feature
            extractor.
        - Training functionality via the DUQ trainer (TrainDUQ).
        - Methods to compute model certainty and OOD scores.

    Upon initialization, the DUQ head and trainer are set up as attributes.

    Attributes:
        model (DUQModel): The DUQ model head.
        trainer (TrainDUQ): Trainer for optimizing the DUQ head.
        device (torch.device): Device for model and trainer computations.
        num_classes (int): Number of in-distribution (ID) classes.
    """

    def __init__(self,
                 feature_extractor: nn.Module,
                 num_classes: int,
                 optimizer: torch.optim.Optimizer,
                 centroid_size: int = 32,
                 sigma: float = 0.1,
                 gamma: float = 0.999,
                 loss_criterion:
                 torch.nn.modules.loss._Loss = F.binary_cross_entropy,
                 device: torch.cuda.device = torch.device(
                    'cuda' if torch.cuda.is_available() else "cpu")
                 ) -> None:
        """
        Initialize the DUQ wrapper.

        Args:
            feature_extractor (nn.Module): Pretrained feature extractor.
                Flattened features should come from the penultimate layer.
            num_classes (int): Number of in-distribution (ID) classes.
            optimizer (Optimizer): Optimizer for training the DUQ head.
            centroid_size (int, optional): Dimensionality of per-class
                centroids. Defaults to 32.
            sigma (float, optional): Initial value for log_sigma shaping the
                RBF kernel. Positivity enforced via softplus. Defaults to 0.1.
            gamma (float, optional): Moving average update rate for centroids.
                New information is added at a rate of 1 - gamma.
                Defaults to 0.999.
            loss_criterion (nn.modules.loss._Loss, optional): Loss function
                for training. Defaults to F.binary_cross_entropy.
            device (torch.device, optional): Device for training and inference
                ('cuda' or 'cpu'). Defaults to CUDA if available.
        """

        self.model = DUQModel(feature_extractor,
                              num_classes,
                              centroid_size=centroid_size,
                              sigma=sigma,
                              gamma=gamma).to(device)
        self.trainer = TrainDUQ(self.model,
                                optimizer=optimizer(
                                    self.model.parameters(),
                                    lr=1e-3,
                                    weight_decay=1e-4),
                                loss_criterion=loss_criterion,
                                n_classes=num_classes,
                                device=device)

    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              n_epochs: int = 100,
              penalty_w: float = 0.05,
              patience: int = 10,
              threshold: None | torch.Tensor = None,
              trial: None | int = None
              ) -> None:
        """
        Executes a training loop for the DUQ model as defined in the
        trainer class. It uses a two-sided gradient penalty and early stopping.
        It sets the found DUQ model head as a class attribute.

        Args:
            train_loader (DataLoader): A PyTorch Dataloader
                object storing data from the train split.
            val_loader (DataLoader): A PyTorch Dataloader
                object storing data from the validation split.
            n_epochs (int, optional): Number of epochs for which to train the
                model head. Defaults to 100.
            penalty_w (float, optional): The magnitude of the gradient
                penalty. Defaults to 0.05.
            patience (int, optional): Defines how long to resume model
                training when the criterion on the validation set does not
                improve. Defaults to 10.
            threshold (None | torch.Tensor, optional):  The thresholds defined
                for the feature extractor when clipping activations using.
                ReAct. Defaults to None.
            trial (None | int, optional): When tuning the hyperparameters
                using Optune this defines which trail it is at.
                Defaults to None.

        Raises:
            optuna.TrialPruned: Prunes model training if the loss is not
                decreasing fast enough as seen in previous trials.
        """
        self.model = self.trainer.train(train_loader,
                                        val_loader,
                                        n_epochs,
                                        penalty_w,
                                        patience,
                                        threshold,
                                        trial)

    def DUQ_ood_score(self,
                      feats: torch.Tensor,
                      **kwargs
                      ) -> np.ndarray:
        """
        Compute model certainty using DUQ.

        Args:
            feats (torch.Tensor): A feature vector.

        Returns:
            np.ndarray: The negative maximum similarity of the sample to any
                of the learned class distributions.
        """
        self.model.eval()
        with torch.no_grad():
            similarities = self.model.feed_features(feats)
        max_similarities = -similarities.max(dim=1).values
        return max_similarities.detach().cpu().numpy()


class TuneDUQModel():
    """
    Hyperparameter tuning class for the DUQ model.

    This class provides utilities to tune the DUQ model's hyperparameters
    with optuna in order to maximize out-of-distribution (OOD) detectability.
    The hyperparameters are centroid size,
    gamma and the magnitude of the two-sided gradient penalty.

    Attributes:
        feature_extractor (nn.Module): Pretrained feature extractor.
        n_classes (int): Number of in-distribution (ID) classes.
        ood_label (int): The class designated as out-of-distribution (OOD).
        data (dict[str, DataLoader]): Dictionary of data splits, typically
            containing 'train', 'val', and 'test' DataLoader objects.
        device (torch.device): Device for computations ('cuda' or 'cpu').
    """
    def __init__(self,
                 feature_extractor: nn.Module,
                 n_classes: int,
                 OOD_label: int,
                 data: dict[DataLoader],
                 device: torch.cuda.device = torch.device(
                    'cuda' if torch.cuda.is_available() else "cpu")
                 ) -> None:
        """
        Initialize the DUQ hyperparameter tuner.

        Args:
            feature_extractor (nn.Module): Pretrained feature extractor.
            n_classes (int): Number of ID classes.
            OOD_label (int): Label of the class marked as OOD.
            data (dict[DataLoader]): Dictionary containing data splits for
                training, validation, and testing. Must include 'train',
                'val', and 'test' keys.
            device (torch.device, optional): Device for training or evaluation.
                Defaults to CUDA if available
        """

        self.ood_label = OOD_label
        self.data = data
        self.device = device
        self.feature_extractor = feature_extractor
        self.n_classes = n_classes

    def _objective(self,
                   trial: int
                   ) -> float:
        """
        The objective function which suggest hyperparameters
        in order to optimize OOD detectability of the DUQ head.

        Args:
            trial (int): The current trail.

        Returns:
            float: The AUROC value on the test set.
        """

        centroid_size = trial.suggest_categorical(
            "centroid_size", [32, 64, 128, 256]
            )
        penalty_w = trial.suggest_float(
            "penalty_w", 1e-5, 5e-2, log=True
            )
        one_minus_gamma = trial.suggest_float(
            "one_minus_gamma", 1e-3, 1e-1, log=True
            )
        gamma = 1.0 - one_minus_gamma

        model = DUQModel(
            self.feature_extractor,
            self.n_classes,
            centroid_size=centroid_size,
            gamma=gamma).to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=1e-3, weight_decay=1e-4
            )
        trainer = TrainDUQ(
            model, optimizer, n_classes=self.n_classes, device=self.device
            )
        trained_model = trainer.train(self.data['train'],
                                      val_loader=self.data['val'],
                                      n_epochs=200,
                                      penalty_w=penalty_w,
                                      trial=trial)
        scores = []
        labels = []

        trained_model.eval()

        with torch.no_grad():
            for X, y in self.data['test']:
                X = X.to(self.device)
                labels.extend(y.detach().cpu().numpy())
                similarities = trained_model(X)
                max_similarities = -similarities.max(dim=1).values
                max_similarities = max_similarities.detach().cpu().numpy()
                scores.extend(max_similarities)
        labels = np.array(labels)
        y_true = (labels == self.ood_label).astype(int)

        auroc = roc_auc_score(y_true, np.array(scores))

        return auroc

    def run_optuna(self,
                   n_trials: int = 30
                   ) -> optuna.study.study:
        """
        Performs the optuna hyperparameter tuning for the DUQ head.
        Dumps the found hyperparameters to a file.

        Args:
            n_trials (int, optional): The number of trails optuna should try
                to find hyperparameters. Defaults to 30.

        Returns:
            optuna.study.study: Returns an optuna study object storing the
                found hyperparameters and the value.
        """
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(
                                        seed=42),
                                    pruner=optuna.pruners.MedianPruner(
                                        n_warmup_steps=5))
        study.optimize(self._objective, n_trials=n_trials)

        print("Best trial: ", study.best_trial.params)
        print("Value (AUROC):", study.best_value)
        print("Params:", study.best_params)

        new_data = {
            "value": study.best_value,
            "params": study.best_params
            }
        try:
            with open("best_trial_1.json", "r") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
        except (FileNotFoundError, json.JSONDecodeError):
            data = []
        data.append(new_data)

        with open("best_trial_1.json", "w") as f:
            json.dump(data, f, indent=2)

        return study
