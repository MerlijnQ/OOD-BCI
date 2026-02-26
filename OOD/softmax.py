import torch
import torch.nn.functional as F
import numpy as np


def softmax_ood_score(logits: torch.Tensor
                      ) -> np.ndarray:
    """
    Returns the maximum softmax score
    for any of the predicted classes in order
    to asses model certainty for a sample.

    Args:
        logits (torch.Tensor): A logit vector(s)
            from the final classification layer of the model.

    Returns:
        np.ndarray: The maximum softmax score for the predicted
            logits for any of the predicted classes.
    """
    probs = F.softmax(logits, dim=1)
    OOD_score, _ = torch.max(probs, dim=1)
    return -OOD_score.detach().cpu().numpy()
