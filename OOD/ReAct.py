import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ReAct():
    """
    Implementation of Rectified Activations (ReAct) for out-of-distribution
    detection.

    ReAct modifies the intermediate activations of a model by
    clamping them based on a specified percentile, as proposed in:
    https://proceedings.neurips.cc/paper_files/paper/2021/file/01894d6f048493d2cacde3c579c315a3-Paper.pdf

    Attributes:
        device (Union[str, torch.device]): Device to perform inference
            ('cpu' or 'cuda').
        percentile (float): Percentile threshold for clamping activations.
    """
    def __init__(self,
                 device: str | torch.cuda.device = 'cuda',
                 percentile: float = 0.90
                 ) -> None:
        """
        Initialize the ReAct object.

        Args:
            device (Union[str, torch.device], optional): Device for model
                inference. Defaults to 'cuda'.
            percentile (float, optional): Determines the magnitude at which
                intermediate activations are clamped. Defaults to 0.90.
        """
        self.percentile = percentile
        self.device = device

    def get_clamp(self,
                  val_loader: DataLoader,
                  model: nn.Module | None = None
                  ) -> torch.Tensor:
        """
        Calculates the thresholds at which the activations get clamped.
        This is determined by the 90th percentile of ID validation data.

        Args:
            val_loader (DataLoader): A PyTorch Dataloader object containing
                ID validation samples.
            model (nn.Module): A trained model on which to perform inference.

        Returns:
            torch.Tensor: A tensor containing the determined thresholds.
        """
        activations = []

        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(self.device)
                feats = model.feature_extractor(x)
                activations.append(feats)

        activations = torch.cat(activations, dim=0)
        max_per_dim = torch.quantile(activations, self.percentile, dim=0)
        return max_per_dim

    def set_clamp(self,
                  x: torch.Tensor,
                  threshold: torch.Tensor
                  ) -> torch.Tensor:
        """
        Clamps the activations of feature vectors according
        to the threshold.

        Args:
            x (torch.Tensor): The feature vectors.
                threshold (torch.Tensor): Determines the maximum activation
                magnitude.

        Returns:
            torch.Tensor: The clamped feature vectors.
        """
        return torch.minimum(x, threshold)

    def get_difference_activations(self,
                                   data_loader: DataLoader,
                                   OOD_label: int,
                                   model: nn.Module
                                   ) -> tuple[float]:
        """
        Determines the difference in activations between
        ID and OOD samples.

        Args:
            data_loader (DataLoader): A PyTorch Dataloader object containing
                both OOD abnd ID samples.
            OOD_label (int): The label of the class that is marked OOD.
                model (nn.Module): A trained model on which to perform
                inference.

        Returns:
            tuple[float]: The n times activations for an OOD samples were
                higher, mean difference, standard deviation, mean max
                difference.
        """

        activations = []
        labels = []
        with torch.no_grad():
            for x, label in data_loader:
                x = x.to(self.device)
                labels.append(label)
                feats = model.feature_extractor(x)
                activations.append(feats)

        activations = torch.cat(activations, dim=0)
        labels = torch.cat(labels, dim=0)

        ood_mask = labels == OOD_label
        id_mask = ~ood_mask

        ood_acts = activations[ood_mask]
        id_acts = activations[id_mask]

        ood_max_1 = ood_acts.max(dim=1).values
        # Max over positive in all dimensions
        id_max_1 = id_acts.max(dim=1).values
        # [N]

        ood_mean = ood_max_1.mean().item()
        id_mean = id_max_1.mean().item()
        diff_1 = ood_mean - id_mean

        ood_max = ood_acts.max(dim=0).values  # [D]
        id_max = id_acts.max(dim=0).values  # [D]

        diff = ood_max - id_max  # [D]
        mean = diff.mean().item()
        std = diff.std().item()

        diff_positive = diff[diff > 0].cpu().numpy()

        n = len(diff_positive)

        return n, mean, std, diff_1
