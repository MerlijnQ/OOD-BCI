import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import regex as re
import argparse


class PlotViolin:
    """
    Class to create violin plots of AUROC results for different models.

    Attributes:
        dataset (str): Name of the dataset for output filenames.
        custom_colors (dict[str, str]): Mapping of model names to colors.
        order (list[str]): Order of models to display in the plot.
    """

    def __init__(self, dataset: str) -> None:
        """
        Initialize PlotViolin with dataset name.

        Args:
            dataset (str): Dataset name.
        """
        self.dataset = dataset
        self.custom_colors = {
            "Distance": "#0072B2",
            "DUQ": "#0072B2",
            "KNN": "#0072B2",
            "Bayesian": "#009E73",
            "MC Dropout": "#009E73",
            "DE": "#009E73",
            "Density": "#E69F00",
            "Energy": "#E69F00",
            "DDU": "#E69F00",
            "DDU EM": "#E69F00",
            "DDU Native": "#E69F00",
            "DDU EM Z": "#E69F00",
            "DDU Native Z": "#E69F00",
            "DDU EM L2": "#E69F00",
            "DDU Native L2": "#E69F00",
            "Softmax": "#8C8C8C",
            "Other": "#CC79A7",
        }
        self.order = ["DE", "MC Dropout", "DDU",
                      "Energy", "DUQ", "KNN", "Softmax"]

    def plot_violin(self, results: dict[list[float]]) -> None:
        """
        Create a violin plot for AUROC results of a single set of models.

        Args:
            results (dict[str, list[float]]): dictionary with model names as
                keys and lists of AUROC scores as values.
        """
        df = pd.DataFrame({
            "Model": [k for k, v in results.items() for _ in v],
            "AUROC": [val for v in results.values() for val in v]
        })

        plt.figure(figsize=(6, 4))
        palette = [self.custom_colors.get(m, "#87CEEB") for m in self.order]

        sns.violinplot(
            x="AUROC",
            y="Model",
            data=df,
            inner="box",
            cut=0,
            order=self.order,
            palette=palette
        )
        sns.despine(top=True, right=True)
        plt.ylabel("")
        plt.xlabel("AUROC ↑")
        plt.xlim(0.0, 1.0)
        plt.xticks(np.arange(0, 1.01, 0.1))
        plt.tight_layout()
        plt.savefig(f"violin_{self.dataset}.pdf", bbox_inches='tight')
        plt.close()

    def plot_violin_comparison(
        self,
        results_a: dict[str, list[float]],
        results_b: dict[str, list[float]],
        labels: tuple[str, str] = ("2 Classes", "3 Classes")
    ) -> None:
        """
        Create a violin plot comparing AUROC results from two sets of models.

        Args:
            results_a (dict[str, list[float]]): First set of results.
            results_b (dict[str, list[float]]): Second set of results.
            labels (tuple[str, str]): Labels for the two sets
            (default: ("2 Classes", "3 Classes")).
        """
        df_a = pd.DataFrame({
            "Model": [f"{k}\n{labels[0]}" for k,
                      v in results_a.items() for _ in v],
            "BaseModel": [k for k, v in results_a.items() for _ in v],
            "AUROC": [val for v in results_a.values() for val in v],
        })

        df_b = pd.DataFrame({
            "Model": [f"{k}\n{labels[1]}" for k,
                      v in results_b.items() for _ in v],
            "BaseModel": [k for k, v in results_b.items() for _ in v],
            "AUROC": [val for v in results_b.values() for val in v],
        })

        df = pd.concat([df_a, df_b], ignore_index=True)

        order = []
        for m in self.order:
            order.extend([f"{m}\n{labels[0]}", f"{m}\n{labels[1]}"])

        palette = []
        for m in self.order:
            c = self.custom_colors.get(m, "#87CEEB")
            palette.extend([c, c])

        plt.figure(figsize=(6, 4))
        ax = sns.violinplot(
            x="AUROC",
            y="Model",
            data=df,
            order=order,
            palette=palette,
            inner="box",
            cut=0,
        )

        for i, body in enumerate(ax.collections):
            if i % 2 == 1:
                body.set_alpha(0.6)
                body.set_hatch("//")

        sns.despine(top=True, right=True)
        plt.ylabel("")
        plt.xlabel("AUROC ↑")
        plt.xlim(0.0, 1.0)
        plt.xticks(np.arange(0, 1.01, 0.1))
        plt.tight_layout()
        plt.savefig(f"violin_{self.dataset}.pdf", bbox_inches="tight")
        plt.close()

    def load(self, file: str) -> None:
        """
        Load results from a JSON file and plot them.

        Args:
            file (str): Path to the JSON file containing results.
        """
        with open(file, "r") as f:
            contents = json.load(f)

        results = contents.get("results")
        if results:
            results = {
                re.sub(r'Deep Ensemble', 'DE', k): v for k,
                v in results.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Create violin plots for AUROC results.")
    parser.add_argument("--dataset",
                        type=str,
                        required=True,
                        help="Dataset name for output files.")
    parser.add_argument("--file_a",
                        type=str,
                        required=True,
                        help="Path to first JSON file with results.")
    parser.add_argument("--file_b",
                        type=str,
                        help="Path to second JSON file for comparison.")
    args = parser.parse_args()

    plotter = PlotViolin(args.dataset)

    results_a = plotter.load(args.file_a)

    if args.file_b:
        with open(args.file_b, "r") as f:
            results_b = json.load(f)
        plotter.plot_violin_comparison(results_a, results_b)
    else:
        plotter.plot_violin(results_a)


if __name__ == "__main__":
    main()
