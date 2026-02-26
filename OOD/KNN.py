import torch
import torch.nn.functional as F
import numpy as np
import faiss
from sklearn.metrics import roc_auc_score
import torch.nn as nn
from OOD.ReAct import ReAct
from torch.utils.data import DataLoader


class KNN():
    """
    K-Nearest Neighbors (KNN) model using FAISS for out-of-distribution (OOD)
    detection.

    This class uses in-distribution (ID) feature vectors to initialize a
    FAISS-based KNN, implementing the OOD detection approach proposed in:
    https://proceedings.mlr.press/v162/sun22d.html

    Attributes:
        max_k (int): Maximum number of neighbors to search during
            hyperparameter selection.
        _KNN_model: Internal KNN model initialized with ID features.
    """
    def __init__(self,
                 ID_features: torch.Tensor,
                 max_k: int
                 ) -> None:
        """
        Initialize the KNN OOD detection model.

        Args:
            ID_features (torch.Tensor): Feature vectors of in-distribution
                samples.
            max_k (int): Maximum allowed k to search during hyperparameter
                selection.
        """
        self.max_k = max_k
        self._KNN_model = self._get_KNN_model(ID_features)

    def _get_KNN_model(self,
                       ID_features: torch.Tensor
                       ) -> faiss.IndexFlatL2:
        """
        L2 normalizes the ID features and fits them to a FAISS model.

        Args:
            ID_features (torch.Tensor): ID feature vectors.

        Returns:
            faiss.IndexFlatL2: An index based FAISS model for KNN.
        """
        ID_np = F.normalize(ID_features, dim=-1).numpy().astype(np.float32)
        index = faiss.IndexFlatL2(ID_np.shape[1])
        index.add(ID_np)
        return index

    def KNN_ood_score(self,
                      z_test: torch.Tensor,
                      k: int = 45,
                      **kwargs: dict
                      ) -> list[float]:
        """
        Computes the distance score between the input tensor and the k-th
        closest neighbor from the ID training set.

        Args:
            z_test (torch.Tensor): Feature vectors.
            k (int, optional): The k-th neighbor from which to return the
                distance score. Defaults to 45.

        Returns:
            list[float]: The distance scores from the samples to their k-th
                nearest neighbor in the ID train set.
        """

        z_test = F.normalize(z_test, dim=-1).cpu().numpy().astype(np.float32)
        D, _ = self._KNN_model.search(z_test, k)
        return D[:, k-1].tolist()

    def tune_k(self,
               model: nn.Module,
               test_loader: DataLoader,
               OOD_label: int,
               device: torch.cuda.device | str,
               use_ReAct: bool,
               r: None | ReAct,
               threshold: None | torch.Tensor = None
               ) -> int:
        """
        Finds the best k hyperparameter value with the optimal OOD
        detectability using AUROC, limited to max_k.

        Args:
            model (nn.Module): A trained model with a function
                feature_extractor() to get the flattened feature vectors
                from the penultimate layer of the model.
            test_loader (DataLoader): A PyTorch Dataloader object containing
                both ID and OOD data.
            OOD_label (int): The class marked as OOD.
            device (torch.cuda.device | str): The device on which model
                inference should be performed.
            use_ReAct (bool): Whether or not to use ractified activations.
            r (None | ReAct): The ReAct object used to clamp the activations.
            threshold (None | torch.Tensor, optional): The threshold by which
                to clamp the activations if ReAct is activated.
                Defaults to None.

        Returns:
            int: The k hyperparameter value that provides the highest AUROC
            score on OOD detection.
        """

        if model.training:
            model.eval()

        all_features = []
        all_labels = []

        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device)
                feats = model.feature_extractor(X)
                if use_ReAct:
                    feats = r.set_clamp(feats, threshold)

                feats = F.normalize(feats, dim=-1
                                    ).cpu().numpy().astype(np.float32)

                all_features.append(feats)
                all_labels.append(y.numpy())

        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        y_true = (all_labels == OOD_label).astype(int)

        best_k = []
        D, _ = self._KNN_model.search(all_features, self.max_k)

        best_k = 1
        best_auroc = 0.0

        for k in range(1, self.max_k + 1):
            kth_dist = D[:, k - 1]
            auroc = roc_auc_score(y_true, kth_dist)

            if auroc > best_auroc:
                best_auroc = auroc
                best_k = k

        return best_k
