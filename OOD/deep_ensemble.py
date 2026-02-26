import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
from model.EEGNeX import EEGNeX_8_32
from model.train import Trainer
from OOD.ReAct import ReAct
from torch.utils.data import DataLoader


class DeepEnsemble():
    """
    Trains and manages an ensemble of EEGNeX models.

    This class initializes `n` ensemble members with different random seeds
    and trains them independently on the provided EEG data. It stores only
    the `state_dict` of each ensemble member to save memory. During inference,
    these states can be loaded into a single EEGNeX model instance.

    On-task performance is recorded on the validation split and stored in
    the `mean_auroc` attribute.

    Attributes:
        data (dict[str, DataLoader]): Dictionary containing data splits
            ('train', 'val', 'test').
        ensemble_states (list[dict[torch.Tensor]]): List of trained model
            states.
        mean_auroc (float | None): Macro on-task AUROC across ensemble members
            on the validation set.
    """
    def __init__(self,
                 data: dict[DataLoader],
                 first_model: None | dict[str, torch.Tensor] = None,
                 criterion:
                 torch.nn.modules.loss._Loss = nn.CrossEntropyLoss(),
                 optimizer: torch.optim.Optimizer = optim.Adam,
                 lr: float = 1e-3,
                 n: int = 5,
                 seeds: list = [17, 77, 123, 314, 42],
                 epochs: int = 200
                 ) -> None:
        """
        Initialize the DeepEnsemble trainer.

        Args:
            data (dict[str, DataLoader]): Dictionary containing data splits.
                Keys should include 'train' and 'val', and 'test' for ID/OOD
                evaluation.
            first_model (dict[torch.Tensor], optional):
                State dictionary of an already trained model.
                Expected to be trained using the last seed in `seeds`.
                Defaults to None.
            criterion (nn.modules.loss._Loss, optional): Loss function to
                optimize. Defaults to nn.CrossEntropyLoss().
            optimizer (torch.optim.Optimizer, optional): Optimizer class for
                training. Defaults to torch.optim.Adam.
            lr (float, optional): Learning rate for optimizer. Defaults to
                1e-3.
            n (int, optional): Number of ensemble members. Defaults to 5.
            seeds (list[int], optional): Unique random seeds for initializing
                ensemble members. Defaults to [17, 77, 123, 314, 42].
            epochs (int, optional): Number of training epochs for each
                ensemble member. Defaults to 200.
        """

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else "cpu")
        self.criterion = criterion
        self._models = []

        self.data = data

        self._model = None
        self.mean_auroc = None

        if first_model is not None:
            state_dict = {k: v.cpu() for k, v in first_model.items()}
            self._models.append(state_dict)
            n = n-1

        self.optimizer = optimizer
        self._train_ensembles(lr=lr, seeds=seeds, n=n, epochs=epochs)

    def _set_seed(self,
                  seed: int = 42
                  ) -> None:
        """
        This function changes the seed which is used for the
        weight initalization of a model and any additional random processes.

        Args:
            seed (int, optional): The desired seed. Defaults to 42.
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _compute_entropy(self,
                         probs: torch.Tensor
                         ) -> np.ndarray:
        """
        Computes the entropy over the provided probabilities.

        Args:
            probs (torch.Tensor): Probabilities with shape [B, C].

        Returns:
            np.ndarray: The per sample entropy [B].
        """
        sum = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        return sum.detach().cpu().numpy()

    def _ensemble_forward(
            self,
            X: torch.Tensor,
            *,
            use_ReAct: bool = False,
            r: None | ReAct = None,
            thresholds: torch.tensor = None,
            ) -> torch.Tensor:
        """
        A forward pass through the ensemble.

        Args:
            X (torch.Tensor): A raw input tensor(s).
            use_ReAct (bool, optional): Whether to apply rectified
                activations according to a specified activation threshold.
                Defaults to False.
            r (None | ReAct, optional): The ReAct object limiting the
                activations according to a threshold. Defaults to None.
            thresholds (torch.tensor, optional): The thresholds defined per
                ensemble member for ReAct. Defaults to None.

        Returns:
            torch.Tensor: The softmax probabilities averaged over the
                ensemble members in shape [B, C].
        """

        outs = []

        for i, model_dict in enumerate(self._models):
            self._model.load_state_dict(
                {k: v.to(self.device) for k, v in model_dict.items()})
            self._model.eval()

            with torch.no_grad():
                if use_ReAct:
                    z = self._model.feature_extractor(X)
                    z = r.set_clamp(z, thresholds[i])
                    out = self._model.classify(z)
                else:
                    out = self._model(X)

                outs.append(out)

        logits = torch.stack(outs, dim=0)  # [M, B, C]

        probs = F.softmax(logits, dim=-1).mean(dim=0)  # [B, C]
        return probs

    def _get_task_auroc(self) -> float:
        """
        Gets the on task (macro) AUROC of the ensemble on a validation set.

        Returns:
            float: The on task (macro) AUROC.
        """
        data_points = []
        labels = []

        with torch.no_grad():
            for X, label in self.data['test_ID']:
                X = X.to(self.device)
                label = label.to(self.device)

                mean_probs = self._ensemble_forward(X)
                data_points.append(mean_probs.cpu())
                labels.append(label.cpu())

        data_points = torch.cat(data_points).numpy()
        labels = torch.cat(labels).numpy()

        n_classes = len(np.unique(labels))

        if n_classes == 2:
            return roc_auc_score(labels, data_points[:, 1])

        y_true_bin = label_binarize(labels, classes=list(range(n_classes)))
        return roc_auc_score(
            y_true_bin,
            data_points,
            average="macro",
            multi_class="ovr",
        )

    def _train_ensembles(self,
                         seeds: list,
                         n: int,
                         lr: float,
                         epochs: int
                         ) -> None:
        """
        Trains an ensemble of EEGNeX models. The states of the
        members are saved as a class attribute.

        Args:
            seeds (list): A list of unique seeds used for weight
                initialization.
            n (int): The number of ensemble members.
            lr (float): Learning rate used by the optimizer
                for model training.
            epochs (int): Number of epochs for which to train a
                model.
        """
        if n > len(seeds):
            raise ValueError(
                "Number of models to train exceeds the number of available \
                seeds")

        info = self.data['info']

        for i in range(n):

            self._set_seed(seeds[i])

            model = EEGNeX_8_32(info['n_classes'],
                                info['n_timesteps'],
                                info['n_channels'],
                                drop_prob=0.5).to(self.device)
            model.train()
            optimizer = self.optimizer(model.parameters(), lr=lr)
            trainer = Trainer(model, optimizer, self.criterion, self.device)
            model = trainer.train(self.data, n_epochs=epochs)
            state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            self._models.append(state_dict)
            del model

        self._model = EEGNeX_8_32(info['n_classes'],
                                  info['n_timesteps'],
                                  info['n_channels'],
                                  drop_prob=0.5).to(self.device)
        self._model.eval()
        self.mean_auroc = self._get_task_auroc()
        self._set_seed(len(seeds)-1)

    def ensemble_ood_score(self,
                           use_ReAct: bool,
                           r: None | ReAct,
                           dataloader: DataLoader
                           ) -> np.ndarray:
        """
        Loops over the test set to calculate
        ensemble entropy in order to asses ensemble certainty.

        Args:
            use_ReAct (bool, optional): Whether to apply rectified
                activations according to a specified activation threshold.
                Defaults to False.
            r (None | ReAct, optional): The ReAct object limiting the
                activations according to a threshold. Defaults to None.
            dataloader (DataLoader): The dataloader to use for computing
                the ensemble OOD score. Contains both ID and OOD samples.

        Returns:
            np.ndarray: The entropy over the predictions made on
                the test set of shape [B].
        """
        self._model.eval()

        thresholds = None
        if use_ReAct:
            thresholds = []
            for model_dict in self._models:
                self._model.load_state_dict(model_dict)
                thresholds.append(
                    r.get_clamp(
                        val_loader=self.data['val'], model=self._model)
                    )

        mean_probs = []

        for x, _ in dataloader:
            x = x.to(self.device)

            probs = self._ensemble_forward(
                x,
                use_ReAct=use_ReAct,
                r=r,
                thresholds=thresholds,
            )
            mean_probs.append(probs)

        all_probs = torch.cat(mean_probs, dim=0)
        return self._compute_entropy(all_probs)
