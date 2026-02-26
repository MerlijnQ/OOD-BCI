import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer():
    """
    Trainer class for supervised training of a PyTorch model with early
    stopping.

    This class provides a training loop following the EEGNeX paper:
    https://doi.org/10.1016/j.bspc.2023.105475

    This includes:
    - Early stopping based on validation performance.
    - Reduce-on-plateau learning rate scheduling with factor 0.5 and a
        patience of 5.

    Attributes:
        model (nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer used for training.
        loss_criterion (nn.modules.loss._Loss): Loss function used to optimize
            the model.
        device (torch.device): Device on which training is performed
            ('cpu' or 'cuda').
        scheduler (torch.optim.lr_scheduler.ReduceLROnPlateau): Scheduler that
            reduces learning rate when validation metric plateaus.
    """
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_criterion:
                 torch.nn.modules.loss._Loss = nn.CrossEntropyLoss(),
                 device: torch.cuda.device = torch.device(
                     'cuda' if torch.cuda.is_available() else 'cpu')
                 ) -> None:
        """
        Initialize the Trainer.

        Args:
            model (nn.Module): The PyTorch model to train.
            optimizer (torch.optim.Optimizer): Optimizer for model training.
            loss_criterion (nn.modules.loss._Loss, optional): Loss function.
                Defaults to nn.CrossEntropyLoss().
            device (torch.device, optional): Device to run training on.
                Defaults to CUDA if available, otherwise CPU.
        """

        self.model = model
        self.loss_criterion = loss_criterion

        self.optimizer = optimizer
        self.device = device
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    def _get_loss(self,
                  X: torch.Tensor,
                  y: torch.Tensor
                  ) -> torch.Tensor:
        """
        Computes the loss according to the criterion
        provided during class initialization (used for
        model optimization).

        Args:
            X (torch.Tensor): A raw input vector.
            y (torch.Tensor): Defines the label belonging to the input.

        Returns:
            torch.Tensor: The model error between the prediction and the label
                as defined by the criterion.
        """
        logits = self.model(X)
        loss = self.loss_criterion(logits, y)
        return loss

    def test(self,
             test_loader: DataLoader
             ) -> float:
        """
        Performs a forward pass on the test/validation split
        of the dataset. This provides an indication on how the model
        would perform on unseen samples. Used for early stopping and
        monitering of the training process.

        Args:
            test_loader (DataLoader): A part of the dataset used for
                intermediate model evaluation during training. These samples
                should not be part of the training dataset.

        Returns:
            float: The loss averaged over the samples in the provided
                test/validation dataset.
        """

        self.model.eval()

        total_n = 0

        with torch.no_grad():
            test_loss = 0.0

            for _, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(
                    self.device), targets.to(
                        self.device)

                loss = self._get_loss(inputs, targets)

                batch_size = inputs.size(0)
                test_loss += loss.item() * batch_size
                total_n += batch_size

            avg_test_loss = test_loss / total_n
            return avg_test_loss

    def train(self,
              data: dict[DataLoader],
              n_epochs: int = 200,
              patience: int = 20
              ) -> nn.Module:
        """
        A loop for model training with early stopping.

        Args:
            data (dict[DataLoader]): A dictionary containing the different
                datasplits. The keys should contain 'train' and 'val' for the
                corresponding splits.
            n_epochs (int, optional): Maximum  number of epochs to train the
                model. Defaults to 200.
            patience (int, optional): Defines how long to resume model
                training when the criterion on the validation set does not
                improve. Defaults to 20.

        Returns:
            nn.Module: Returns the model with the lowest validation loss
                found during the training process.
        """

        best_model_weights = None
        train_loader, val_loader = data['train'], data['val']

        best_val_loss = float('inf')
        no_improve = 0

        self.model.conv5.apply_spectral()

        for epoch in range(n_epochs):
            self.model.train()
            for X, y in train_loader:
                X, y = X.to(
                    self.device), y.to(
                        self.device)

                self.optimizer.zero_grad()
                loss = self._get_loss(X, y)
                loss.backward()
                self.optimizer.step()

            val_loss = self.test(val_loader)
            self.scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = self.model.state_dict()
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                print('Early stopping at epoch {}'.format(epoch))
                break

        if best_model_weights:
            self.model.load_state_dict(best_model_weights)
            print('Loaded best model weights.')

        self.model.conv5.remove_spectral()
        return self.model
