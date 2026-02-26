import torch
import numpy as np


class DDU():
    """
    Deep Deterministic Uncertainty (DDU).

    Implements the DDU approach for out-of-distribution (OOD) detection
    based on feature vectors from a pretrained model's penultimate layer,
    as proposed by:
    https://doi.org/10.1109/CVPR52729.2023.02336

    Attributes:
        _gmm_model: The Gaussian Mixture Model used for DDU-based OOD scoring.
    """
    def __init__(self,
                 ID_features: torch.Tensor,
                 device: str,
                 n_classes: int,
                 ID_labels: np.ndarray
                 ) -> None:
        """
        Initialize the DDU model.

        Args:
            ID_features (torch.Tensor): Feature vectors from the penultimate
                layer of a pretrained model.
            device (str): Device for constructing the DDU model
                ('cpu' or 'cuda'). Must match the device of ID_features.
            n_classes (int): Number of in-distribution (ID) classes.
            ID_labels (np.ndarray): Array indicating which features belong to
                each class.
        """

        self._gmm_model = self._get_DDU_model(
            ID_features, device, n_classes, ID_labels)

    def _centered_cov_torch(self,
                            x: torch.Tensor
                            ) -> torch.Tensor:
        """
        Centers the coveriance of classes.
        From the official DDU implementation:
        https://github.com/omegafragger/DDU

        Args:
            x (torch.Tensor): Feature vectors per class
                centered around their class mean.

        Returns:
            torch.Tensor: The centered covariance matrix.
        """
        n = x.shape[0]
        res = 1 / (n - 1) * x.t().mm(x)
        return res

    def _get_DDU_model(self,
                       ID_features: torch.Tensor,
                       device: str,
                       n_classes: int,
                       ID_labels: np.ndarray
                       ) -> torch.distributions.MultivariateNormal:
        """
        Fits a per class Gaussian on the ID feature vectors
        constucting a gaussian mixture model. Adapted from
        the official DDU implementation:
        https://github.com/omegafragger/DDU

        Args:
            ID_features (torch.Tensor): Feature vectors from the penultimate
            layer of the model on which the GMM is fitted.
            device (str): The device on which the DDU model nees to be
            constructed. Needs to match the device type of the input tensor.
            n_classes (int): Number of ID classes it needs to fit.
            ID_labels (np.ndarray): An array defining which feature vectors
            belong to which class.

        Raises:
            RuntimeError: An error raised if with the defined jitter values
                a positive definite covariance could not be created.

        Returns:
            torch.distributions.MultivariateNormal: A gaussian mixture model
                defined in the model feature space.
        """
        DOUBLE_INFO = torch.finfo(torch.double)
        jitter_eps_list = [0, DOUBLE_INFO.tiny] + [
            10 ** exp for exp in range(-308, 0, 1)]
        labels = torch.tensor(ID_labels, dtype=torch.long)

        means = torch.stack([torch.mean(ID_features[labels == c],
                                        dim=0) for c in range(n_classes)])

        classwise_cov_features = torch.stack(
                    [self._centered_cov_torch(
                        ID_features[labels == c] - means[c]
                        ) for c in range(n_classes)])

        gmm = None
        for jitter_eps in jitter_eps_list:
            try:
                jitter = jitter_eps * torch.eye(
                            classwise_cov_features.shape[1],
                            device=classwise_cov_features.device,
                        ).unsqueeze(0)
                cov = classwise_cov_features + jitter
                gmm = torch.distributions.MultivariateNormal(
                    loc=means.to(device), covariance_matrix=cov.to(device),
                )
                print(f"Selected jitter_eps: {jitter_eps}")
                break
            except (RuntimeError, ValueError) as e:
                if (
                    "cholesky" in str(e)) or (
                        "covariance_matrix" in str(e)) or (
                            "PositiveDefinite" in str(e)) or (
                                "The parameter covariance_matrix has invalid \
                                    values" in str(e)):
                    continue

        if gmm is None:
            raise RuntimeError(
                "Could not create positive definite covariance even \
                after trying jitters.")

        return gmm

    def DDU_ood_score(self, z_test: torch.Tensor) -> np.ndarray:
        """
        Calculates the probability that the input feature vector
        is part of any of the per class distributions that were fitted on the
        ID data.

        Args:
            z_test (torch.Tensor): A feature vector belonging to an unknown
                sample.

        Returns:
            np.ndarray: The model certainty that the sample belongs to the
                ID distribution.
        """
        probs = self._gmm_model.log_prob(z_test[:, None, :])  # [N, C]
        probs = torch.logsumexp(probs, dim=1)  # Marginalize over classes [N]
        return -probs.detach().cpu().numpy()
