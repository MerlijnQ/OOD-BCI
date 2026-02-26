import numpy as np
import matplotlib.pyplot as plt
import regex as re
import json
import pandas as pd
from scipy.stats import gaussian_kde
from typing import Dict, Any
import argparse


class PlotDelta:
    """
    Class to plot delta distributions of AUROC differences between two
    conditions.

    Attributes:
        dataset (str): Dataset name for output filenames.
        order (list[str]): Order of models to plot.
    """

    def __init__(self, dataset: str) -> None:
        """
        Initialize the PlotDelta object.

        Args:
            dataset (str): Name of the dataset.
        """
        self.dataset = dataset
        self.order = ["DE", "MC Dropout", "DDU",
                      "Energy", "DUQ", "KNN", "Softmax"]

    def plot_delta(self,
                   df: pd.DataFrame,
                   df_react: pd.DataFrame,
                   inverse: bool = False
                   ) -> None:
        """
        Plot the distribution of delta AUROC values between two DataFrames.

        Args:
            df (pd.DataFrame): Base AUROC values.
            df_react (pd.DataFrame): AUROC values under ReAct or Inversion
            condition.
            inverse (bool): If True, plots delta for Inversion-2classes; else
            ReAct-noReAct.
        """
        diff = {key: np.array(
            df_react[key]) - np.array(df[key]) for key in df.keys()}

        plt.figure(figsize=(10, 6))
        okabe_ito_7 = ["#000000",
                       "#E69F00",
                       "#56B4E9",
                       "#009E73",
                       "#F0E442",
                       "#0072B2",
                       "#D55E00"]

        for i, key in enumerate(self.order):
            if key not in diff:
                raise ValueError(f"Key '{key}' not found in diff dictionary.")
            values = diff[key]
            kde = gaussian_kde(values)
            xs = np.linspace(values.min(), values.max(), 300)

            plt.plot(
                xs,
                kde(xs),
                color=okabe_ito_7[i % len(okabe_ito_7)],
                linewidth=2.5,
                label=key,
            )

        plt.axvline(0, color="black", linestyle="--", linewidth=1.3)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.ylim(bottom=0, top=20)
        plt.xlim(left=-0.5, right=0.5)
        plt.xticks(np.arange(-0.5, 0.6, 0.25))

        if inverse:
            plt.xlabel("Δ (Inversion - 2 classes)", fontsize=18)
        else:
            plt.xlabel("Δ (ReAct - no ReAct)", fontsize=18)
            plt.legend(fontsize=18)

        plt.ylabel("Density", fontsize=18)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

        plt.tight_layout()
        plt.savefig(f"deltas_react_{self.dataset}.pdf", bbox_inches="tight")
        plt.close()

    def load(self, file: str) -> Dict[str, Any]:
        """
        Load results from a JSON file and rename 'Deep Ensemble' to 'DE'.

        Args:
            file (str): Path to JSON file.

        Returns:
            Dict[str, Any]: Dictionary of processed results.
        """
        with open(file, "r") as f:
            results = json.load(f).get("results")
        results = {re.sub(r'Deep Ensemble', 'DE', k): v for k,
                   v in results.items()}
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Plot delta AUROC distributions from a JSON file.")
    parser.add_argument("--dataset",
                        type=str,
                        required=True,
                        help="Dataset name for output files.")
    parser.add_argument("--file",
                        type=str,
                        required=True,
                        help="Path to JSON file containing results.")
    parser.add_argument("--file_ReAct",
                        type=str,
                        required=True,
                        help="Path to JSON file containing ReAct or inverse results.")
    parser.add_argument("--inverse",
                        action="store_true",
                        help="Plot delta for Inversion insetad of ReAct.")
    args = parser.parse_args()

    plotter = PlotDelta(args.dataset)
    results = plotter.load(args.file)
    results_react = plotter.load(args.file_ReAct)

    df = pd.DataFrame({k: v for k, v in results.items()})
    df_react = pd.DataFrame({k: v for k, v in results_react.items()})

    plotter.plot_delta(df, df_react, inverse=args.inverse)


if __name__ == "__main__":
    main()
