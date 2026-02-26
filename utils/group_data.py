import json
import argparse


def group_results(dataset: str = 'Schirrmeister2017',
                  n: int = 14,
                  react: bool = False,
                  inverse: bool = False
                  ) -> None:
    """
    Groups the results for n participants into a single file.

    Args:
        dataset (str, optional): Name of the dataset used in the experiment.
            Defaults to 'Schirrmeister2017'.
        n (int, optional): Number of subjects for which results were generated.
            Defaults to 14.
        react (bool, optional): Whether ReAct was activated in the experiment.
            Defaults to False.
        inverse (bool, optional): Whether the experiment was done with
            selective inversion. Defaults to False.
    """

    subjects = list(range(2, n+1))

    def double(auroc: list) -> list:
        """
        Duplicates on-task auroc score. Used for the inversion experiment,
        where two subsequent OOD detection scores belong to the same model.

        Args:
            auroc (list): On-task auroc scores that need to be duplicated.

        Returns:
            List: Duplicated auroc scores.
        """
        new = []
        for value in auroc:
            new.append(value)
            new.append(value)
        return new

    all_results = {}
    task_performance = {}
    for subject in subjects:
        with open(
                rf"results_{dataset}_{subject}.json",
                "r") as f:
            data = json.load(f)
            results = data.get("results")
            if subject == 2:
                all_results = results
                auc = data.get("val_auroc")
                if len(auc) < 12 and len(auc) != 4:
                    task_performance["val_auroc"] = double(
                        data.get("val_auroc"))
                    task_performance["duq_val_auroc"] = double(
                        data.get("duq_val_auroc"))
                else:
                    task_performance["val_auroc"] = data.get(
                        "val_auroc")
                    task_performance["duq_val_auroc"] = data.get(
                        "duq_val_auroc")
                task_performance["deep_ensamble_auroc"] = data.get(
                    "deep_ensamble_auroc")
            else:
                for key, value in results.items():
                    all_results[key].extend(value)
                auc = data.get("val_auroc")
                if len(auc) < 12 and len(auc) != 4:
                    task_performance["val_auroc"].extend(
                        double(data.get("val_auroc")))
                    task_performance["duq_val_auroc"].extend(
                        double(data.get("duq_val_auroc")))
                else:
                    task_performance["val_auroc"].extend(data.get(
                        "val_auroc"))
                    task_performance["duq_val_auroc"].extend(data.get(
                        "duq_val_auroc"))

    r = {"results": all_results,
         "performance": task_performance}

    print("yes")
    if inverse:
        file = f'results_inverse_baseline_{dataset}.json'
    else:
        file = f'results_{dataset}.json'

    with open(file, "w") as f:
        json.dump(r, f, indent=4, sort_keys=True)

    if inverse:
        all_results = {}
        task_performance = {}
        for subject in subjects:
            with open(
                    rf"results_inversion_{dataset}_{subject}.json",
                    "r") as f:
                data = json.load(f)
                results = data.get("results")
                if subject == 2:
                    all_results = results
                else:
                    for key, value in results.items():
                        all_results[key].extend(value)

        r = {"results": all_results}

        with open(f'results_inverse_{dataset}.json', "w") as f:
            json.dump(r, f, indent=4, sort_keys=True)

    if react:
        all_results = {}
        react_data = {}
        for subject in subjects:
            with open(
                    f'results_ReAct_{dataset}_{subject}.json',
                    "r") as f:

                data = json.load(f)
                results = data.get("results")
                if subject == 2:
                    all_results = results
                    react_data["N"] = data.get("differences").get("N")
                    react_data["M"] = data.get("differences").get("M")
                    react_data["std"] = data.get("differences").get("std")
                    react_data["Mean Max Activation difference"] = data.get(
                        "differences").get("Mean Max Activation difference")
                else:
                    for key, value in results.items():
                        all_results[key].extend(value)
                        react_data["N"].extend(data.get("differences").get(
                            "N"))
                        react_data["M"].extend(data.get("differences").get(
                            "M"))
                        react_data["std"].extend(data.get("differences").get(
                            "std"))
                        react_data["Mean Max Activation difference"].extend(
                            data.get("differences").get(
                                "Mean Max Activation difference"))

        r = {"results": all_results,
             "react_data": react_data}
        with open(f'results_ReAct_{dataset}.json', "w") as f:
            json.dump(r, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Group experiment results into a single file."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="Schirrmeister2017",
        help="Name of the dataset used in the experiment."
    )

    parser.add_argument(
        "--n",
        type=int,
        default=14,
        help="Number of subjects."
    )

    parser.add_argument(
        "--react",
        action="store_true",
        help="Activate ReAct processing."
    )

    parser.add_argument(
        "--inverse",
        action="store_true",
        help="Activate inversion experiment processing."
    )

    args = parser.parse_args()

    group_results(
        dataset=args.dataset,
        n=args.n,
        react=args.react,
        inverse=args.inverse
    )
