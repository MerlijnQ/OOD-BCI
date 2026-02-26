import torch
import numpy as np


def energy_score(logits: torch.Tensor
                 ) -> np.ndarray:
    """
    Calculates the energy score for a given input tensor
    in order to assess model certainty on the predicted tensor.
    This is class inveriant. Proposed by:
    https://proceedings.neurips.cc/paper_files/paper/2020/file/f5496252609c43eb8a3d147ab9b9c006-Paper.pdf

    Args:
        logits (torch.Tensor): A logit vector(s)
            from the final classification layer of the model.

    Returns:
        np.ndarray: The energy score for the predicted logits.
    """
    scores = torch.logsumexp(logits, dim=-1)
    return -scores.detach().cpu().numpy()
