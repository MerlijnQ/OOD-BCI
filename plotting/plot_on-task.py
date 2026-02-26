import os
import json
import regex as re
import matplotlib.pyplot as plt
import argparse


class Plot:
    """
    Class for plotting OOD detection results against validation AUROC.

    Attributes:
        dataset (str): Name of the dataset for naming the output files.
    """

    def __init__(self, dataset: str):
        """
        Initialize the Plot object with a dataset name.

        Args:
            dataset (str): Name of the dataset.
        """
        self.dataset = dataset

    def _plot_scatter(
        self,
        auroc_val: list[float],
        auroc_ood_a: list[float],
        auroc_ood_b: list[float],
        methods: list[str],
        output_dir: str,
        deep_ensemble_auroc: list[float] | None = None,
        duq_auroc: list[float] | None = None,
    ) -> None:
        """
        Plot a scatter plot comparing validation AUROC and OOD AUROC.

        Args:
            auroc_val (list[float]): Validation AUROC scores.
            auroc_ood_a (list[float]): OOD AUROC scores for the first method.
            auroc_ood_b (list[float]): OOD AUROC scores for the second method.
            methods (list[str]): Names of the methods to be plotted.
            output_dir (str): Directory where the plot PDF will be saved.
            deep_ensemble_auroc (Optional[list[float]]): Deep Ensemble AUROC
            scores, if any.
            duq_auroc (Optional[list[float]]): DUQ AUROC scores, if any.
        """
        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(8, 6))

        plt.scatter(auroc_val, auroc_ood_a, color='blue', label=methods[0])

        if deep_ensemble_auroc is not None:
            plt.scatter(deep_ensemble_auroc,
                        auroc_ood_b,
                        color="green",
                        label=methods[1])
        elif duq_auroc is not None:
            plt.scatter(duq_auroc,
                        auroc_ood_b,
                        color="green",
                        label=methods[1])
        else:
            plt.scatter(auroc_val,
                        auroc_ood_b,
                        color="green",
                        label=methods[1])

        plt.xlabel('On-Task Macro AUROC', fontsize=16)
        plt.ylabel('OOD AUROC', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(False)
        plt.legend(fontsize=16, loc='upper left')
        plt.tight_layout()

        filename = os.path.join(
            output_dir,
            f"{self.dataset}_results_{methods[0]}_{methods[1]}.pdf")
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

    def _plot_OOD_vs_val(
        self,
        val_auroc: list[float],
        deep_ensemble_auroc: list[float],
        ood_results: dict[str, list[float]],
        duq_auroc: list[float],
        output_dir: str = "scatter_results",
    ) -> None:
        """
        Create scatter plots comparing OOD AUROC scores with validation AUROC.

        Args:
            val_auroc (list[float]): Validation AUROC scores.
            deep_ensemble_auroc (list[float]): Deep Ensemble AUROC scores.
            ood_results (dict[str, list[float]]): dictionary of OOD AUROC
            results keyed by method.
            duq_auroc (list[float]): DUQ AUROC scores.
            output_dir (str): Directory to save plots.
        """
        self._plot_scatter(
            val_auroc,
            ood_results.get("KNN"),
            ood_results.get("DUQ"),
            methods=["KNN", "DUQ"],
            duq_auroc=duq_auroc,
            output_dir=output_dir,
        )
        self._plot_scatter(
            val_auroc,
            ood_results.get("MC Dropout"),
            ood_results.get("Deep Ensemble"),
            methods=["MC Dropout", "Deep Ensemble"],
            output_dir=output_dir,
            deep_ensemble_auroc=deep_ensemble_auroc,
        )
        self._plot_scatter(
            val_auroc,
            ood_results.get("Energy"),
            ood_results.get("DDU"),
            methods=["Energy", "DDU"],
            output_dir=output_dir,
        )

    def load_and_plot(self, file: str) -> None:
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

        deep_ensemble_auroc = contents.get("emsemble_auroc")
        duq_auroc = contents.get("duq_auroc")
        val_auroc = contents.get("val_auroc")

        self._plot_OOD_vs_val(
            val_auroc=val_auroc,
            deep_ensemble_auroc=deep_ensemble_auroc,
            ood_results=results,
            duq_auroc=duq_auroc,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Plot OOD AUROC results from a JSON file.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (used in output filenames)"
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the JSON file containing results"
    )
    args = parser.parse_args()

    plotter = Plot(args.dataset)
    plotter.load_and_plot(args.file)


if __name__ == "__main__":
    main()
