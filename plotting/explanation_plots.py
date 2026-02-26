import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
from sklearn.manifold import TSNE
import torch.nn.functional as F
from scipy.interpolate import RBFInterpolator
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import torch.nn as nn
from torch.utils.data import DataLoader


def normalizer(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10)


class ExplPlots():
    """Class integrating utils for plotting visual explanations
    for Bayesian, Distance and Density based OOD detection methods.

    Attributes:
        dataset (str): Name of the dataset used.
        device (dataLoader): Device on which to perform inference.
            Defaults to CUDA if available, else cpu.
        class_colors: Colors selected to portray different classes.
        class_labels_map: Mapping class integers to labels. Default is
            {0: 'Left hand', 1: 'Right hand'}.
        fig_size: Size of the generated figures.
        dot_size: Size of the samples plotted in the figures. Defaults
            to 250.
        legend_fontsize: Font size of text in the legend if present in the
            Figure.

    """
    def __init__(self, dataset_name: str) -> None:
        """Initializes the plotting class.

        Args:
            dataset_name (str): Name of the dataset of which the data is used.
        """
        self.dataset = dataset_name
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else "cpu")

        self.class_colors = {0: '#440154', 1: "#E6CD00"}
        self.ood_color = '#D55E00'
        self.class_labels_map = {0: 'Left hand', 1: 'Right hand'}

        self.fig_size = (10, 8)
        self.dot_size = 250
        self.legend_fontsize = 14

    def _reduce_samples(self,
                        labels: np.ndarray,
                        reduced: np.ndarray,
                        samples: int = 10
                        ) -> dict:
        """
        Selects a specified number of samples that are
        closest to the mean sample of the distribution of samples.

        Args:
            labels (np.ndarray): Lablels belonging to samples.
            reduced (np.ndarray): Samples which to pick from.
            samples (int, optional): Number of samples to sample.
                Defaults to 10.

        Returns:
            dict: The selected samples per label category.
        """
        selected_indices = {}
        unique_classes = np.unique(labels)
        for cls in unique_classes:
            cls_indices = np.where(labels == cls)[0]
            cls_points = reduced[cls_indices]
            cls_centroid = np.mean(cls_points, axis=0)
            dists = np.linalg.norm(cls_points - cls_centroid, axis=1)
            closest_idx = cls_indices[np.argsort(dists)[:samples]]
            selected_indices[cls] = closest_idx
        return selected_indices

    def _plot_selected(self,
                       features: np.ndarray,
                       selected_indices: dict,
                       name: str,
                       OOD_label: None | int = None,
                       show_legend: bool = False,
                       knn_k: int = 3,
                       max_ood_arrows: int = 10
                       ) -> None:
        """
        Plots a 2D visualization of selected feature points with arrows.

        This function generates a scatter plot of the provided feature vectors,
        coloring points according to their class. If an OOD 
        class is specified, arrows are drawn from OOD points to their nearest 
        ID neighbors, up to a maximum number of arrows. The 
        resulting plot can optionally include a legend and is saved to a file.

        Args:
            features (np.ndarray): 2D array of shape (n_samples, 2) containing
                the feature coordinates to plot.
            selected_indices (dict): A dictionary mapping class labels (int) to
                arrays of indices corresponding to the points of that class in
                `features`.
            name (str): Filename (or relative path) to save the plot.
            OOD_label (None | int, optional): Label corresponding to the OOD
                class. Arrows will be drawn from these points to their nearest
                ID neighbors. Defaults to None (no OOD arrows are drawn).
            show_legend (bool, optional): Whether to display a legend showing
                class names and colors. Defaults to False.
            knn_k (int, optional): Number of nearest neighbors to use when
                drawing arrows from OOD points to ID points. Defaults to 3.
            max_ood_arrows (int, optional): Maximum number of OOD points to
                display with arrows. If there are more OOD points than this
                value, a random subset is selected. Defaults to 10.

        Returns:
            None: The plot is saved to the file specified by 'name'.
        """
        plt.figure(figsize=self.fig_size)
        ax = plt.gca()
        ax.set_facecolor('#fbf8f3')

        # Plot all points
        for cls, idxs in selected_indices.items():
            color = self.ood_color if cls == OOD_label else self.class_colors[
                cls]
            ax.scatter(features[idxs, 0],
                       features[idxs, 1],
                       s=self.dot_size,
                       color=color)

        if OOD_label is not None and OOD_label in selected_indices:
            ood_indices = selected_indices[OOD_label]
            id_indices = np.concatenate(
                [selected_indices[
                    cls] for cls in selected_indices if cls != OOD_label])
            if len(ood_indices) > max_ood_arrows:
                ood_indices = np.random.choice(
                    ood_indices, max_ood_arrows, replace=False)

            knn = NearestNeighbors(n_neighbors=knn_k)
            knn.fit(features[id_indices])

            for ood_idx in ood_indices:
                ood_point = features[ood_idx].reshape(1, -1)
                _, neighbor_idx = knn.kneighbors(ood_point)
                neighbor_point = features[id_indices[neighbor_idx[0][knn_k-1]]]

                ax.annotate('', xy=neighbor_point, xytext=ood_point[0],
                            arrowprops=dict(facecolor='red',
                                            edgecolor='red',
                                            arrowstyle='->',
                                            lw=1.5,
                                            alpha=0.8))
        if show_legend:
            labels, colors = [], []
            for cls in selected_indices:
                labels.append(
                    'OOD' if cls == OOD_label else self.class_labels_map[
                        cls])
                colors.append(
                    self.ood_color if cls == OOD_label else self.class_colors[
                        cls])
            for lbl, col in zip(labels, colors):
                ax.scatter([], [], color=col, s=self.dot_size, label=lbl)
            ax.legend(loc='upper right', fontsize=self.legend_fontsize)

        plt.xticks([])
        plt.yticks([])
        plt.xlabel('')
        plt.ylabel('')
        plt.grid(False)
        plt.savefig(os.path.join("exp", name))
        plt.close()

    def _get_train_features(self,
                            model: nn.Module,
                            data: dict[DataLoader]
                            ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the features produced by the model from the training data.

        Args:
            model (nn.Module): A trained PyTorch model.
            data (dict[DataLoader]): Contains dataloaders belonging to
                different splits including 'train'.

        Returns:
            tuple[np.ndarray, np.ndarray]: Feature vectors and labels.
        """
        ID_features = []
        labels = []
        model.eval()
        with torch.no_grad():
            for X, y in data.get('train'):
                X, y = X.to(self.device), y.to(self.device)
                out = model.feature_extractor(X)
                ID_features.extend(out.detach().cpu())
                labels.extend(y.detach().cpu().numpy())

        ID_features = torch.stack(ID_features).numpy()  # shape: [N, 10]
        ID_features = normalizer(np.array(ID_features))
        labels = np.array(labels)
        return ID_features, labels

    def plot_distance(self,
                      model: nn.Module,
                      data: dict
                      ) -> None:
        """
        Plots TSNE and PCA reduces features to demonstrate their
        seperability by distance in the feature space.

        Args:
            model (nn.Module): A trained PyTorch model.
            data (dict[DataLoader]): Contains dataloaders belonging to
                different splits including 'train'.
        """
        ID_features, labels = self._get_train_features(model, data)

        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        logits_tsne = tsne.fit_transform(ID_features)

        pca = PCA(n_components=2)
        logits_pca = pca.fit_transform(ID_features)

        indeces = self._reduce_samples(labels, logits_tsne)
        self._plot_selected(
            logits_tsne, indeces, name=f"{self.dataset}_tsne_distance.pdf")

        indeces = self._reduce_samples(labels, logits_pca)
        self._plot_selected(
            logits_pca, indeces, name=f"{self.dataset}_pca_distance.pdf")

        OOD_feature, label = self._get_feature_label(model, data)

        OOD_label = len(np.unique(label)) - 1
        OOD_feature = OOD_feature[label == OOD_label]
        OOD_labels = label[label == OOD_label]

        conc = np.concatenate((ID_features, OOD_feature))
        conc_label = np.concatenate((labels, OOD_labels))

        con_pca = pca.fit_transform(conc)
        con_tsne = tsne.fit_transform(conc)

        indeces = self._reduce_samples(conc_label, con_pca)
        self._plot_selected(con_pca,
                            indeces,
                            name=f"{self.dataset}_pca_distance2.pdf",
                            OOD_label=OOD_label)

        indeces = self._reduce_samples(conc_label, con_tsne)
        self._plot_selected(con_tsne,
                            indeces,
                            name=f"{self.dataset}_tsne_distance2.pdf",
                            OOD_label=OOD_label)

    def _plotting_b(self,
                    all_features: np.ndarray,
                    all_pred: np.ndarray,
                    all_labels: np.ndarray,
                    name: str,
                    top_k: int = 10
                    ) -> None:
        """
        Creates a 2D contour plot showing the predicted probability of the
        'Right hand' class based on the top confident samples from two classes.

        This function selects the `top_k` most certain samples per class
        (based on the maximum predicted probability), interpolates a
        probability surface using an RBF interpolator, and plots a filled
        contour map. The top samples are overlaid as scatter points with
        class-specific colors and optional edge colors for visibility.

        Args:
            all_features (np.ndarray): Array containing the feature
                coordinates for all samples.
            all_pred (np.ndarray): Array of predicted probabilities. The mean
                across models is used to compute certainties.
            all_labels (np.ndarray): Array of ground-truth labels.
            name (str): Filename to save the resulting plot.
            top_k (int, optional): Number of most certain samples per class to
                select for plotting and interpolation. Defaults to 10.

        Returns:
            None: The plot is saved to a file.
        """

        all_labels = all_labels[0].numpy()
        mean_probs = all_pred.mean(dim=0).numpy()
        certainties = mean_probs.max(axis=1)

        top_indices = []
        for cls in [0, 1]:
            cls_idx = np.where(all_labels == cls)[0]
            top = cls_idx[np.argsort(certainties[cls_idx])[-top_k:]]
            top_indices.append(top)
        top_indices = np.concatenate(top_indices)

        filtered_features = all_features[top_indices]
        filtered_probs = mean_probs[top_indices]
        filtered_labels = all_labels[top_indices]

        prob_class1 = filtered_probs[:, 1]
        x_vals, y_vals = filtered_features[:, 0], filtered_features[:, 1]

        grid_x, grid_y = np.meshgrid(
            np.linspace(
                all_features[:, 0].min() - 1,
                all_features[:, 0].max() + 1, 300),
            np.linspace(
                all_features[:, 1].min() - 1,
                all_features[:, 1].max() + 1, 300)
        )
        grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        rbf = RBFInterpolator(
            filtered_features, prob_class1, neighbors=min(
                len(top_indices), 30), smoothing=0.1)
        grid_prob = np.clip(
            rbf(grid_points).reshape(grid_x.shape), 0, 1)

        _, ax = plt.subplots(figsize=self.fig_size, constrained_layout=True)
        contour = ax.contourf(
            grid_x, grid_y, grid_prob, levels=20, cmap='viridis', alpha=1.0
            )

        for cls in [0, 1]:
            c = 'white' if cls == 0 else 'black'
            mask = filtered_labels == cls
            ax.scatter(x_vals[mask],
                       y_vals[mask],
                       c=self.class_colors[cls],
                       s=self.dot_size,
                       label=self.class_labels_map[cls],
                       edgecolor=c,
                       linewidths=2.5)

        padding = 0.5

        x_min, x_max = x_vals.min() - padding, x_vals.max() + padding
        y_min, y_max = y_vals.min() - padding, y_vals.max() + padding

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        plt.xticks([])
        plt.yticks([])
        plt.xlabel('')
        plt.ylabel('')

        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('P(Right hand)', fontsize=18)
        cbar.ax.tick_params(labelsize=16)

        plt.savefig(os.path.join("exp", name), dpi=300)
        plt.close()

    def plot_bayesian(self,
                      models: list[nn.Module.state_dict],
                      data: dict[DataLoader]
                      ) -> None:
        """Plots a visual explanation of a Bayesian approach to
        uncertainty estimation using a Deep Ensemble.

        Args:
            models (list[nn.Module.state_dict]): Multiple trained models
                belonging to the ensemble.
            data (dict[DataLoader]): Contains dataloaders belonging to
                different splits including 'train'.
        """

        all_features, all_labels, all_pred = [], [], []

        with torch.no_grad():
            for i, model in enumerate(models):
                model.eval()
                features, predictions, labels = [], [], []
                for X, y in data.get('train'):
                    X, y = X.to(self.device), y.to(self.device)
                    if i == 0:
                        labels.append(y.detach().cpu())

                    logits = model.feature_extractor(X)
                    features.append(logits.cpu())

                    pred = model.classify(logits)
                    softmax = F.softmax(pred, dim=1)
                    predictions.append(softmax.cpu())

                if i == 0:
                    all_labels.append(torch.cat(labels))
                all_features.append(torch.cat(features))
                all_pred.append(torch.cat(predictions))

        all_features = torch.stack(all_features).mean(dim=0).numpy()
        all_features_tsne = TSNE(
            n_components=2, perplexity=30, random_state=0
            ).fit_transform(all_features)
        all_features_pca = PCA(n_components=2).fit_transform(all_features)
        all_pred = torch.stack(all_pred)

        self._plotting_b(all_features_tsne,
                         all_pred,
                         all_labels,
                         f"{self.dataset}_bayesian_tsne.pdf")
        self._plotting_b(all_features_pca,
                         all_pred,
                         all_labels,
                         f"{self.dataset}_bayesian_pca.pdf")

    def _get_feature_label(self,
                           model: nn.Module,
                           data: dict[DataLoader]
                           ) -> tuple[np.ndarray, np.ndarray]:
        """Gets feature vectors from the test set with their
        corresponding labels.

        Args:
            model (nn.Module): A trained PyTorch model.
            data (dict[DataLoader]): Contains dataloaders belonging to
                different splits including 'train'.

        Returns:
            tuple[np.ndarray, np.ndarray]: The computed feature vectors and
                labels.
        """
        features = []
        labels = []

        with torch.no_grad():
            for X, y in data.get('test'):
                X, y = X.to(self.device), y.to(self.device)
                logit = model.feature_extractor(X).cpu().numpy()
                features.extend(logit)
                labels.extend(y.detach().cpu().numpy())

        features = normalizer(np.array(features))
        return np.array(features), np.array(labels)

    def _plot_ellipse(self,
                      ax: matplotlib.axes.Axes,
                      mean: float,
                      cov: np.ndarray,
                      n_std: float = 2.0,
                      facecolor: str = 'white',
                      edgecolor: str = 'black',
                      alpha: float = 0.9
                      ) -> None:
        """
        Plots a covariance ellipse representing a Gaussian distribution.

        Args:
            ax (matplotlib.axes.Axes): Matplotlib axis on which the ellipse
                will be drawn.
            mean (np.ndarray): Mean vector representing the
                center of the Gaussian distribution in 2D space.
            cov (np.ndarray): 2x2 covariance matrix defining the spread and
                orientation of the distribution.
            n_std (float, optional): Number of standard deviations used to
                scale the ellipse radii. Defaults to 2.0.
            facecolor (str, optional): Fill color of the ellipse.
                Defaults to 'white'.
            edgecolor (str, optional): Edge color of the ellipse.
                Defaults to 'black'.
            alpha (float, optional): Transparency of the ellipse.
                Defaults to 0.9.

        Returns:
            None: The ellipse patch is added directly to the provided axis.
        """

        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * n_std * np.sqrt(vals)
        ell = Ellipse(xy=mean,
                      width=width,
                      height=height,
                      angle=theta,
                      facecolor=facecolor,
                      edgecolor=edgecolor,
                      alpha=alpha,
                      linewidth=2.5,)
        ax.add_patch(ell)

    def _plot_DDU(self,
                  model,
                  comp,
                  data: dict,
                  name: str):
        """
        Visualizes class-wise feature densities for Deep Deterministic
        Uncertainty (DDU) using a 2D projection.

        Args:
            model: Trained model used to extract penultimate-layer features.
            comp: A dimensionality reduction object implementing
                'fit_transform' (e.g., PCA, t-SNE, UMAP).
            data (dict): Dictionary containing dataset splits ('train',
                'val', 'test') used to extract ID and OOD samples.
            name (str): Name used to save the resulting density plot.

        Returns:
            None: The figure is saved to a file.
        """
        ID_logits, labels = self._get_train_features(model, data)

        OOD_logit, label = self._get_feature_label(model, data)

        OOD_label = len(np.unique(label)) - 1
        OOD_logits = OOD_logit[label == OOD_label]
        OOD_labels = label[label == OOD_label]

        conc = np.concatenate((ID_logits, OOD_logits))
        conc_label = np.concatenate((labels, OOD_labels))

        reduced = comp.fit_transform(conc)

        indeces = self._reduce_samples(conc_label, reduced, samples=20)

        plot_stats = {}
        for cls, idxs in indeces.items():
            if cls == OOD_label:
                continue
            cls_points = reduced[idxs]
            mean = np.mean(cls_points, axis=0)
            cov = np.cov(cls_points, rowvar=False)
            plot_stats[cls] = (mean, cov)

        plt.figure(figsize=self.fig_size)
        ax = plt.gca()
        ax.set_facecolor('#fbf8f3')

        for cls, idxs in indeces.items():
            if cls == OOD_label:
                continue
            cls_points = reduced[idxs]
            plt.scatter(cls_points[:, 0], cls_points[:, 1],
                        s=self.dot_size,
                        color=self.class_colors[cls]
                        )
            self._plot_ellipse(ax,
                               *plot_stats[cls],
                               n_std=3,
                               edgecolor=self.class_colors[cls])

        max_ood_points = 10
        ood_indices = np.where(conc_label == OOD_label)[0]
        if len(ood_indices) > max_ood_points:
            ood_indices = np.random.choice(ood_indices,
                                           max_ood_points,
                                           replace=False)
        plt.scatter(reduced[ood_indices, 0], reduced[ood_indices, 1],
                    s=self.dot_size * 1.5, color=self.ood_color)
        plt.scatter(reduced[ood_indices, 0], reduced[ood_indices, 1],
                    s=self.dot_size,
                    color=self.ood_color)

        plt.xticks([])
        plt.yticks([])
        plt.xlabel('')
        plt.ylabel('')

        plt.tight_layout()
        plt.savefig(os.path.join("exp", f"{self.dataset}_density_{name}.pdf"))
        plt.close()

    def plot_DDU_visualization(self,
                               model: nn.Module,
                               data: dict,
                               ) -> None:
        """Plots a visualization of DDU by drawing ellipses according to a
        gaussian distribution around the samples in a 2D feature space.

        Args:
            model (nn.Module): A trained PyTorch model.
            data (dict[DataLoader]): Contains dataloaders belonging to
                different splits including 'train'.
        """
        pca = PCA(n_components=2)
        self._plot_DDU(model, pca, data, name='pca')
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        self._plot_DDU(model, tsne, data, name='tsne')
