import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import regex as re
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse


class PlotHeatmaps:
    """
    Class to create heatmaps for AUROC results across subjects, classes,
    and methods.

    Attributes:
        dataset (str): Dataset name for output filenames.
        order (list[str]): Default order of models for plotting.
    """

    def __init__(self, dataset: str) -> None:
        """
        Initialize the PlotHeatmaps object.

        Args:
            dataset (str): Name of the dataset.
        """
        self.dataset = dataset
        self.order = ["DE", "MC Dropout", "DDU",
                      "Energy", "DUQ", "KNN", "Softmax"]

    def heatmap_SD(self, results: dict[list[float]]) -> None:
        """
        Plot a heatmap of standard deviations per subject for each model.

        Args:
            results (dict[list[float]]): dictionary of model results.
        """
        std = {}
        n = 4  # number of classes
        for key, value in results.items():
            sublists = [value[i:i+n] for i in range(0, len(value), n)]
            stdlist = [np.std(np.array(lst)) for lst in sublists]
            std[key] = stdlist

        df_heatmap = pd.DataFrame(std).T
        df_heatmap = df_heatmap.loc[[
            x for x in self.order if x in df_heatmap.index]]

        df_heatmap['Mean'] = df_heatmap.mean(axis=1)

        n_subjects = len(df_heatmap.columns) - 1
        subject_col = [f"{i}" for i in range(2, n_subjects+2)]
        df_heatmap.columns = subject_col + ['Mean']

        plt.figure(figsize=(10, 4))
        sns.heatmap(
            df_heatmap,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            cbar_kws={'label': 'SD'},
            annot_kws={"color": "white", "size": 8}
        )
        plt.ylabel("")
        plt.xlabel("Subject")
        plt.tight_layout()
        plt.savefig(f"heatmap_SD_{self.dataset}.pdf", bbox_inches='tight')
        plt.close()

    def heatmap_subjects(self,
                         results: dict[list[float]],
                         n_subjects: int = 60
                         ) -> None:
        """
        Plot a heatmap of mean AUROC per class for each subject.

        Args:
            results (dict[list[float]]): dictionary of model results.
            n_subjects (int): Number of subjects.
        """
        classes = ["Left Hand", "Right Hand", "Feet", "Tongue"]
        n_classes = len(classes)

        agg = {subj: {cls: [] for cls in classes} for subj in range(
            n_subjects)}

        for _, values in results.items():
            for i, v in enumerate(values):
                subj = i // n_classes
                cls = classes[i % n_classes]
                agg[subj][cls].append(v)

        aggregated = {
            subj: {cls: float(np.mean(vals)) for cls, vals in cls_dict.items()}
            for subj, cls_dict in agg.items()
        }

        df = pd.DataFrame(aggregated)
        df['Mean'] = df.mean(axis=1)

        n_subjects = len(df.columns) - 1
        subject_col = [f"{i}" for i in range(2, n_subjects+2)]
        df.columns = subject_col + ['Mean']

        plt.figure(figsize=(10, 4))
        sns.heatmap(
            df,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            cbar_kws={'label': 'Mean AUROC'},
            annot_kws={"color": "white", "size": 8}
        )
        plt.ylabel("Class")
        plt.xlabel("Subject")
        plt.tight_layout()
        plt.savefig(f"heatmap_subjects_{self.dataset}.pdf",
                    bbox_inches='tight')
        plt.close()

    def heatmap_methods(
        self,
        results: dict[list[float]],
        show_xticks: bool = False,
        show_cbar: bool = False,
        cbar_bottom: bool = True
    ) -> None:
        """
        Plot a heatmap showing mean AUROC per class for each method.

        Args:
            results (dict[list[float]]): dictionary of model results.
            show_xticks (bool): Whether to display x-axis labels.
            show_cbar (bool): Whether to display a colorbar.
            cbar_bottom (bool): If True, colorbar is at bottom; else right.
        """
        classes = ["Right", "Left", "Rest", "Feet"]
        desired_class_order = ["Right", "Left", "Feet", "Rest"]
        n_classes = len(classes)

        agg = {model: {cls: [] for cls in classes} for model in results.keys()}
        for model, values in results.items():
            for i, v in enumerate(values):
                cls = classes[i % n_classes]
                agg[model][cls].append(v)

        aggregated = {
            model: {cls: float(np.mean(vals)) for cls,
                    vals in cls_dict.items()}
            for model, cls_dict in agg.items()
        }

        df = pd.DataFrame(aggregated)
        ordered_cols = [m for m in self.order if m in df.columns]
        df = df.reindex(columns=ordered_cols)
        df = df.reindex(desired_class_order)
        df['Mean'] = df.mean(axis=1)

        vmin = 0.328
        vmax = 0.750

        if show_xticks:
            fig, ax = plt.subplots(figsize=(10, 6.5))
        else:
            fig, ax = plt.subplots(figsize=(10, 5))

        ax = sns.heatmap(
            df,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            square=False,
            annot_kws={"size": 14},
            vmin=vmin,
            vmax=vmax,
            cbar=False
        )

        if show_cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes(
                "bottom" if cbar_bottom else "right",
                size="6%",
                pad=0.5
            )

            cbar = fig.colorbar(
                ax.collections[0],
                cax=cax,
                orientation="horizontal" if cbar_bottom else "vertical"
            )
            cbar.set_label("Mean AUROC", fontsize=18)
            cbar.ax.tick_params(labelsize=16)

        for text in ax.texts:
            value = float(text.get_text())
            normalized = (value - vmin) / (vmax - vmin)
            text.set_color("white" if normalized < 0.5 else "black")

        plt.xticks(rotation=30, ha='right')
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)

        if not show_xticks:
            ax.set_xticklabels([])
            ax.set_xlabel("")

        plt.tight_layout()
        plt.savefig(f"heatmap_models_{self.dataset}.pdf", bbox_inches='tight')
        plt.close()

    def load(self, file: str) -> dict[list[float]]:
        """
        Load results from a JSON file and rename 'Deep Ensemble' to 'DE'.

        Args:
            file (str): Path to JSON file.

        Returns:
            dict[str, list[float]]: Processed results dictionary.
        """
        with open(file, "r") as f:
            results = json.load(f).get("results")
        results = {re.sub(r'Deep Ensemble', 'DE', k): v for k,
                   v in results.items()}
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Create heatmaps for AUROC results.")
    parser.add_argument("--dataset",
                        type=str,
                        required=True,
                        help="Dataset name for output files.")
    parser.add_argument("--file",
                        type=str,
                        required=True,
                        help="Path to JSON file with results.")
    args = parser.parse_args()

    plotter = PlotHeatmaps(args.dataset)
    results = plotter.load(args.file)

    plotter.heatmap_SD(results)
    plotter.heatmap_subjects(results)
    plotter.heatmap_methods(results)


if __name__ == "__main__":
    main()
