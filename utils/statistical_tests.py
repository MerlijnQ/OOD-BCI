
import json
import numpy as np
import scipy.stats as stats
import scikit_posthocs as sp
import pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrix
import scipy.stats as st
from statsmodels.stats.multitest import multipletests
from itertools import combinations
from tabulate import tabulate


class statistical_tests():
    def _filter_category(self,
                         data: dict
                         ) -> tuple:
        """Group the results based on category.

        Args:
            data (dict): Results from the experiment loaded from a file.

        Returns:
            tuple: Method labels and corresponding matrix with results.
        """
        labels = ["Bayesian", "Density", "Distance", "Softmax"]
        matrix = []
        matrix.append(
            np.mean([data.get("MC Dropout"), data.get("Deep Ensemble")],
                    axis=0))
        matrix.append(
            np.mean([data.get("MC Dropout"), data.get("Energy")],
                    axis=0))
        matrix.append(
            np.mean([data.get("KNN"), data.get("DUQ")],
                    axis=0))
        matrix.append(
            data.get("Softmax"))
        return labels, matrix

    def _filter_normally(self,
                         data: dict
                         ) -> tuple:
        """Create a matrix from the loaded resulsts.
        Args:
            data (dict): Results from the experiment loaded from a file.

        Returns:
            tuple: Method labels and corresponding matrix with results.
        """
        matrix = []
        labels = []
        for key in data.keys():
            matrix.append(data[key])
            labels.append(key)
        return labels, matrix

    def load_data(self, paths: str | list,
                  cat: bool = False,
                  val_auroc: bool = False
                  ) -> pd.DataFrame:
        """
        Loads the experiment results for a dataset into a dataframe
        that can be used for statistical tests.

        Args:
            paths (str | list): The path(s) to a file(s) storing the
                experiment results.
            cat (bool, optional): Whether to group categories.
                Defaults to False.
            val_auroc (bool, optional): Whether to load on task performance
                metrics. Defaults to False.

        Raises:
            TypeError: When path is not of the expected type.

        Returns:
            pd.DataFrame: A dataframe with the data in a format that
            can be used for the statistical tests.
        """

        if isinstance(paths, str):
            paths = [paths]

        OOD_results = None
        valid_auroc = None
        duq_auroc = None
        deep_auroc = None

        for file_path in paths:
            if not isinstance(file_path, str):
                raise TypeError("expected file path as a string")

            with open(file_path, "r") as f:
                data = json.load(f)

            if OOD_results is None:
                OOD_results = data
                if val_auroc:
                    valid_auroc = OOD_results.get("val_auroc")
                    deep_auroc = OOD_results.get("deep_ensamble_auroc")
                    duq_auroc = OOD_results.get("duq_val_auroc")
                    print(len(valid_auroc), len(duq_auroc), len(deep_auroc))

            else:
                for key in OOD_results["results"]:
                    OOD_results["results"][key].extend(data["results"][key])
                    print(len(OOD_results["results"][key]))

                if val_auroc:
                    valid_auroc.extend(data.get("val_auroc"))
                    deep_auroc.extend(data.get("deep_ensamble_auroc"))
                    duq_auroc.extend(data.get("duq_val_auroc"))

        if cat:
            labels, matrix = self._filter_category(OOD_results.get('results'))
        else:
            labels, matrix = self._filter_normally(OOD_results.get('results'))
            print("Loaded data for", labels)

            matrix = np.array(matrix).T
            df = pd.DataFrame(matrix, columns=labels)

            if val_auroc:
                print(len(valid_auroc), len(duq_auroc), len(deep_auroc))
                return df, valid_auroc, duq_auroc, deep_auroc

        return df

    def spearmans(self,
                  results: pd.DataFrame,
                  val_auroc: pd.DataFrame,
                  deep_auroc: pd.DataFrame,
                  duq_auroc: pd.DataFrame
                  ) -> None:
        """
        Computes spearman correlation between on-task performance
        and OOD detectability. DUQ and Deep ensembles have their own
        trainable paramaters which is accounted for here.

        Args:
            results (pd.DataFrame): Dataset resulting from an experiment loaded
                using this class load function into the right format.
            val_auroc (pd.DataFrame): On-task performance.
            deep_auroc (pd.DataFrame): On-task performance Deep ensembles.
            duq_auroc (pd.DataFrame): On-task performance DUQ.
        """

        df = pd.DataFrame(columns=["p", "corr"])
        for method in results.columns:
            if method == "Deep Ensemble":
                corr, p = stats.spearmanr(deep_auroc, results[method].values)
            if method == "DUQ":
                corr, p = stats.spearmanr(duq_auroc, results[method].values)
            else:
                corr, p = stats.spearmanr(val_auroc, results[method].values)
            df.loc[method] = [p, corr]

        print(df)

    def IQR(self, results: pd.DataFrame) -> None:
        """
        Computes Median and IQR per methods for a provided dataset.

        Args:
            results (pd.DataFrame): Dataset resulting from an experiment loaded
                using this class load function into the right format.
        """
        df = pd.DataFrame(columns=['Mdn', 'IQR'])
        for method in results.columns:
            std = stats.iqr(results[method].values)
            mdn = np.median(results[method].values)
            df.loc[method] = [mdn, std]
        print(df)

    def kruskal(self, path: str, class_names: list) -> None:
        """
        kruskal wallis test used in order to determine differences
        between detectability of different OOD classes.

        Args:
            path (str): Path to a file with results.
            class_names (list): The classes used in the experiment as OOD in
                the same order that they were marked as OOD in the experiment.

        Raises:
            ValueError: If an uncompatible number of classes is provided.
        """
        df = self.load_data(paths=path)
        num_classes = len(class_names)

        combined = []
        for col in df.columns:
            combined.extend(df[col].dropna().tolist())

        combined = np.array(combined)

        if len(combined) % num_classes != 0:
            raise ValueError(
                f"AUROC list length ({len(combined)}) must be divisible by \
                      number of classes ({num_classes})."
            )

        num_runs = len(combined) // num_classes
        reshaped = combined.reshape((num_runs, num_classes))

        scores = reshaped.flatten()
        class_labels = np.tile(class_names, num_runs)

        df_long = pd.DataFrame({'AUROC': scores, 'Class': class_labels})

        summary = df_long.groupby('Class')['AUROC'].agg(
            ['median', lambda x: np.median(x)]
            )
        summary.columns = ['Median', 'IQR']
        print(summary)

        grouped = df_long.groupby('Class')['AUROC'].apply(list)
        H, p = stats.kruskal(*grouped)
        print("H-statistic:", H)
        print("p-value:", p)

        if p < 0.05:
            posthoc = sp.posthoc_dunn(df_long,
                                      val_col='AUROC',
                                      group_col='Class',
                                      p_adjust='bonferroni')
            print("\nDunn posthoc p-values:")
            print(posthoc)
        else:
            print("No significant differences between classes (p ≥ 0.05).")

    def wilcoxen(self,
                 dataset: dict,
                 dataset_other: dict
                 ) -> None:
        """
        Wilcoxon signed rank test. Used to determine the effect of
        ReAct and selective inversion. Results have to be of the same length.

        Args:
            dataset (dict): Results on one experiment.
            dataset_other (dict): Results on another experiment that is
                hypothesised to improve over the other one.
        """

        baseline = self.load_data(dataset)
        other = self.load_data(dataset_other)
        df = pd.DataFrame(columns=["p", "effect size"])

        for method in baseline.columns:
            stat, p = stats.wilcoxon(
                other[method].values, baseline[method].values,
                alternative="greater"
                )
            df.loc[method] = [p, stat]

        print(df)

    def whitney(self, dataset: dict,
                dataset_other: dict
                ) -> None:
        """
        Performs a Mann-Whitney U Test which can be used
        to compare results from models trained on a different
        number of classes. Results can have a lenght difference.

        Args:
            dataset (dict): Results on one experiment.
            dataset_other (dict): Results on another experiment.
        """

        baseline = self.load_data(dataset)
        other = self.load_data(dataset_other)
        df = pd.DataFrame(columns=["p", "effect size"])

        for method in baseline.columns:
            stat, p = stats.mannwhitneyu(other[method].values,
                                         baseline[method].values,
                                         alternative="two-sided")
            df.loc[method] = [p, stat]

        print(df)

    def medians(self, dataset: dict) -> None:
        df = self.load_data(dataset)
        print(df.median())

    def react_mean_max_diff(self, model_diffs: list) -> None:
        """
        Performs bootstrappping on the computed mean maximum difference
        between OOD and ID samples to compute if in general ID or OOD have
        higher activations in the penultimate layer.

        Args:
            model_diffs (list): A list of mean max activation differences.
        """

        model_diffs = np.array(model_diffs)
        N = len(model_diffs)

        mean_diff = model_diffs.mean()
        std_diff = model_diffs.std(ddof=1)

        n_boot = 10000
        boot_means = []
        for _ in range(n_boot):
            sample = np.random.choice(model_diffs, size=N, replace=True)
            boot_means.append(sample.mean())
        boot_means = np.array(boot_means)
        ci_lower = np.percentile(boot_means, 2.5)
        ci_upper = np.percentile(boot_means, 97.5)

        print(f"Mean of mean-max differences: {mean_diff:.3f}")
        print(f"Standard deviation: {std_diff:.3f}")
        print(f"95% CI for the mean: [{ci_lower:.3f}, {ci_upper:.3f}]")


class LMM(statistical_tests):
    def __init__(self) -> None:
        super().__init__()
        """
        Initialize a class to perform, an LMM analysis and
        a pairwise estimated means comparison.
        """

    def load_data_LMM(
            self,
            file_path: str,
            dataset_name: str = "BNCI2014",
            cat: bool = False,
            val_auroc: bool = False,
            ) -> pd.DataFrame:
        """
        Load the data from a file with results.

        Args:
            file_path (str): Path to the file on which a test needs to be ran.
            dataset_name (str, optional): Name of the dataset used in the
                experiment. Defaults to "BNCI2014".
            cat (bool, optional): Whether to group categories.
                Defaults to False.
            val_auroc (bool, optional): Whether to load on task performance
                metrics. Defaults to False.

        Returns:
            pd.DataFrame: Data loaded in the format needed for the LMM.
        """

        if val_auroc:
            df_wide, val_auroc_vals, _, deep_auroc_vals = self.load_data(
                file_path,
                cat=cat,
                val_auroc=True
            )
        else:
            df_wide = self.load_data(
                file_path,
                cat=cat,
                val_auroc=False
            )

        labels = df_wide.columns.tolist()

        df_wide["Participant"] = [
            f"{dataset_name}_P{i // 4:02d}" for i in range(len(df_wide))
        ]
        df_wide["Dataset"] = dataset_name

        # Long format for LMM
        df_long = df_wide.melt(
            id_vars=["Participant", "Dataset"],
            value_vars=labels,
            var_name="Method",
            value_name="AUROC"
        )

        if val_auroc:
            return df_long, val_auroc_vals, deep_auroc_vals

        print(df_long['Participant'].nunique())
        print(df_long.groupby("Participant")["AUROC"].var())
        return df_long

    def _emmeans_pairwise(self,
                          df: pd.DataFrame,
                          lmm_result: smf.mixedlm
                          ) -> None:
        """
        Performs a pairwise comparison between methods using the
        marginal estimated means. Adapted from:
        https://glennwilliams.me/blog/posts/estimating-marginal-means-manually-in-python/

        Args:
            df (pd.DataFrame): The data to be analysed.
            lmm_result (smf.mixedlm): The fitted LMM which will be used to
                compute the estimated marginal means.
        """
        grid = pd.DataFrame(
                np.array(np.meshgrid(
                    df['Method'].cat.categories,
                    df['Dataset'].unique()
                )).reshape(2, -1).T,
                columns=['Method', 'Dataset']
                )

        mat = dmatrix(
            "C(Method, Treatment(reference='Softmax')) * C(Dataset)",
            grid,
            return_type="matrix"
        )

        betas = lmm_result.fe_params.values
        emmeans = mat @ betas

        grid['EMM'] = emmeans

        k_fe = len(lmm_result.fe_params)
        vcov = lmm_result.cov_params().iloc[:k_fe, :k_fe].values
        print(np.linalg.matrix_rank(vcov), vcov.shape, np.min(np.diag(vcov)))

        method_agg = grid.groupby('Method', as_index=False).agg(
            EMM=('EMM', 'mean'),
            SE=('EMM', lambda x: np.sqrt(np.mean((x - x.mean())**2)))
        )

        overall_mat = []
        method_order = method_agg['Method'].tolist()
        for method in method_order:
            rows = mat[grid['Method'] == method, :]
            overall_mat.append(np.mean(rows, axis=0))
        overall_mat = np.vstack(overall_mat)

        idx = list(range(len(method_order)))
        pair_indices = list(combinations(idx, 2))

        contrast_mat = (overall_mat[[i for i,
                                     _ in pair_indices], :] -
                        overall_mat[[j for _,
                                     j in pair_indices], :]).T

        contrast_est = betas @ contrast_mat
        contrast_se = np.sqrt(np.maximum(
            [c @ vcov @ c.T for c in contrast_mat.T],
            1e-12
        ))

        print("Min contrast SE:", np.min(contrast_se))
        print("Zero SE contrasts:", np.where(contrast_se < 1e-10))

        z = contrast_est / contrast_se
        p = 2 * st.norm.sf(np.abs(z))
        _, p_adj, _, _ = multipletests(p, method="holm")

        contrast_labels = [
            f"{method_order[i]} - {method_order[j]}" for i,
            j in pair_indices
        ]

        pairwise_table = pd.DataFrame({
            "Contrast": contrast_labels,
            "Estimate": contrast_est,
            "SE": contrast_se,
            "z": z,
            "p": p,
            "p_adj": p_adj
        })

        pairwise_table[['p', 'p_adj']] = pairwise_table[
            ['p', 'p_adj']].applymap(lambda x: f"{x:.3e}")

        print(
            tabulate(
                pairwise_table,
                headers="keys",
                tablefmt="fancy_grid",
                showindex=False,
                colalign=("left", "right", "right", "right", "right", "right")
            )
        )

        fe_table = pd.DataFrame({
            'Coef': lmm_result.fe_params,
            'SE': lmm_result.bse_fe,
            'z': lmm_result.tvalues,
            'p': lmm_result.pvalues
            })
        fe_table_fmt = fe_table.copy()
        fe_table_fmt['p'] = fe_table_fmt['p'].apply(lambda x: f"{x:.3e}")

        print(fe_table_fmt.round(3).to_string())
        print(lmm_result.summary())

    def lmm(self, datasets: dict) -> None:
        """
        Fits a LMM to the data.

        Args:
            datasets (dict): A dict containing the paths
                to all the files (one per dataset) that are used in the
                analysis.
        """

        df_list = []
        for name, path in datasets.items():
            df_list.append(self.load_data_LMM(path, dataset_name=name))
        full_df = pd.concat(df_list, ignore_index=True)

        # Convert Method to categorical and set Softmax as baseline
        full_df['Method'] = pd.Categorical(full_df['Method'])
        all_methods = full_df['Method'].cat.categories.tolist()
        if 'Softmax' in all_methods:
            all_methods.insert(
                0, all_methods.pop(
                    all_methods.index('Softmax')
                    )
                )
        full_df['Method'] = full_df['Method'].cat.reorder_categories(
            all_methods)

        model = smf.mixedlm("AUROC ~ Method * Dataset",
                            data=full_df,
                            groups=full_df["Participant"])
        result = model.fit()

        self._emmeans_pairwise(full_df, result)


def main():
    pass

    # classes = {
    #     "Schirrmeister2017": ["Right hand", "Left hand", "Rest", "Feet"],
    #     "BNCI2014": ["Left hand", "Right hand", "Feet", "Tongue"],
    #     "Stieger2021": ['Right hand', 'Left hand', 'Both hands', 'Rest'],
    #     "inverse": [
    #         "Right hand + Left hand",
    #         "Right hand + Rest",
    #         "Right hand + Feet",
    #         "Left hand + Right hand",
    #         "Left hand + Rest",
    #         "Left hand + Feet",
    #         "Rest + Right hand",
    #         "Rest + Left hand",
    #         "Rest + Feet",
    #         "Feet + Right hand",
    #         "Feet + Left hand",
    #         "Feet + Rest"
    #     ]
    # }


if __name__ == "__main__":
    main()
