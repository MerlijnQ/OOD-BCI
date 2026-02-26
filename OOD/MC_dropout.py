import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from OOD.ReAct import ReAct


def compute_entropy(probs: torch.Tensor
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


def MC_ood_score(test_loader: DataLoader,
                 model: nn.Module,
                 MC_t: int = 50,
                 threshold: torch.Tensor | None = None,
                 r: ReAct | None = None,
                 device: torch.cuda.device = torch.device(
                     'cuda' if torch.cuda.is_available() else 'cpu'),
                 **kwargs
                 ) -> np.ndarray:
    """
    This implementes the MC-dropout based OOD detection as proposed by:
    https://proceedings.mlr.press/v48/gal16.html

    We put the model in training mode such that random dropout is enabled
    to get an estimated distribution of possible predictions for a sample on
    which we compute the entropy as a uncertainty measure.

    Args:
        test_loader (DataLoader): A PyTorch Dataloader object containing
            both ID and OOD data.
        model (nn.Module): A trained PyTorch model (e.g. EEGNeX).
        MC_t (int, optional): The number of forward passes per sample.
            Defaults to 50.
        threshold (torch.Tensor | None): The threshold to apply to the
            activations when using ReAct.
        r (ReAct | None): The ReAct class enforcing activation limits when
            applicable.
        device (torch.cuda.device): The device on which
            model training is performed. Defaults to 'cuda' if
            available else 'cpu'.

    Returns:
        np.ndarray: The entropy over the predictions made on
        the test set of shape [B].
    """
    model.train()

    entropy_scores = []

    with torch.no_grad():
        for X, _ in test_loader:
            X = X.to(device)
            mc_samples = []
            for _ in range(MC_t):
                feat = model.feature_extractor(X)

                if threshold is not None:
                    feat = r.set_clamp(feat, threshold)

                logits = model.classify(feat)
                probs = F.softmax(logits, dim=1)
                mc_samples.append(probs)

            mc_samples = torch.stack(mc_samples)
            mean_prob = mc_samples.mean(dim=0)
            entropy_scores.extend(compute_entropy(mean_prob))

    model.eval()
    return np.array(entropy_scores)
