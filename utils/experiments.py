import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import os
from itertools import combinations
import optuna

from model.EEGNeX import EEGNeX_8_32
from utils.dataLoader import dataLoader
from model.train import Trainer
from OOD.DUQ import TuneDUQModel
from OOD.OOD_pipeline import OOD
from OOD.deep_ensemble import DeepEnsemble
from hyperparameters.hyperparameters import hyperparam
from plotting.explanation_plots import ExplPlots


def set_seed(seed: int = 42) -> None:
    """Setting a seed in order to control randomness.

    Args:
        seed (int, optional): Defaults to 42.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # For single GPU
    torch.cuda.manual_seed_all(seed)  # For multi-GPU


class Experiments():
    """
    Class to run Leave-One-Class-Out experiments for evaluating
    out-of-distribution (OOD) detection on BCI datasets.

    This class provides functionality for:
        - Running the main leave-one-class-out OOD detection experiments.
        - Conducting ReAct and inversion experiments.
        - Hyperparameter tuning pipelines for DUQ and KNN methods.
        - Generating visual explanations of the defined classes/categories.

    Attributes:
        dataset_name (str): Name of the dataset used.
        data_loader (dataLoader): Data loader object for batching and
            accessing dataset.
        subject_list (List[str]): List of subjects in the dataset.
        class_keys (List[str]): List of class labels in the dataset.
        len_classes (int): Number of classes in the dataset.
        device (torch.device): Device used for model training and inference
            (CPU or CUDA).
        criterion (torch.nn.modules.loss._Loss): Loss function used for
            training the base model.
        methods (List[str]): List of OOD detection methods included in the
            experiments.
        OOD_label (int): Label index considered as out-of-distribution for
            leave-one-class-out.
        OOD_label_2 (int | None): Optional secondary OOD label
            (initialized as None).
    """
    def __init__(self,
                 criterion:
                 torch.nn.modules.loss._Loss = nn.CrossEntropyLoss(),
                 dataset: str = "Schirrmeister2017",
                 batch_size: int = 32
                 ) -> None:
        """
        Initialize the Experiments object for leave-one-class-out OOD
        evaluation.

        Args:
            criterion (torch.nn.modules.loss._Loss, optional): Loss function
                for training the base model.
                Defaults to nn.CrossEntropyLoss().
            dataset (str, optional): Name of the dataset to use. Defaults to
                "Schirrmeister2017".
            batch_size (int, optional): Batch size for the data loader.
                Defaults to 32.
        """

        set_seed()

        self.dataset_name = dataset
        self.data_loader = dataLoader(dataset=dataset, batch_size=batch_size)
        self.subject_list = self.data_loader.return_subjects()
        classes = self.data_loader.get_classes()
        self.class_keys = list(classes.keys())
        self.len_classes = len(classes)
        print("Class_keys: {}".format(self.class_keys))

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else "cpu")
        self.criterion = criterion

        self.methods = ["Softmax", "Energy", "DDU", "KNN",
                        "DUQ", "MC Dropout", "Deep Ensemble"]

        self.OOD_label = len(classes) - 1
        self.OOD_label_2 = None

    def _reset_init(self) -> None:
        """Resets the class attributes.
        """
        self.OOD_pipeline, self.model, = None, None
        self.data = {}

        self.results = defaultdict(list)
        self.results_ReAct = defaultdict(list)
        self.act_diff = defaultdict(list)
        self.inversion = defaultdict(list)
        self.inversed = defaultdict(lambda: 0)

        self.val_auroc = []
        self.duq_auroc = []
        self.all_ens_auroc = []

    def get_explanation(self) -> None:
        """
        Creates a figure in order to explain bayesian,
        density and distance based uncertainty estimation.
        """

        p = ExplPlots(self.dataset_name)
        os.makedirs("exp", exist_ok=True)

        subject_n = 8
        class_key = 'feet'
        subject = self.subject_list[subject_n]

        self._get_data_and_model(
            subject, class_key, discard='tongue', model=True
            )
        filename = os.path.join("exp", 'visualization_model.pth')
        torch.save(self.model.state_dict(), filename)
        p.plot_DDU_visualization(self.model, self.data)

        p.plot_distance(self.model, self.data)

        os.makedirs("exp_ens", exist_ok=True)
        DE = DeepEnsemble(self.data, self.model.state_dict())
        models, _, _ = DE.train_ensembles(n=5)
        for i, model in enumerate(models):
            filename = os.path.join("exp_ens", f'ens_model_weights_{i}.pth')
            torch.save(model, filename)

        model_list = []

        info = self.data['info']
        for model in models:
            model_inst = EEGNeX_8_32(info['n_classes'],
                                     info['n_timesteps'],
                                     info['n_channels'],
                                     drop_prob=0.5).to(self.device)
            model_inst.load_state_dict(model)
            model_list.append(model_inst)

        p.plot_bayesian(model_list, self.data)

    def _OOD_parameter_tuning(self
                              ) -> tuple[int, optuna.study.study]:
        """
        Tunes hyperparameters for DUQ and KNN for OOD detection.

        Returns:
            tuple[int, optuna.study.study]: Best K and an optuna study
                containing the found centroid size, 1-gamma and gradient
                penalty weight for DUQ.
        """
        self.OOD_pipeline = OOD(model=self.model,
                                device=self.device,
                                data=self.data,
                                OOD_label=self.OOD_label,
                                methods=self.methods)

        k = self.OOD_pipeline.get_score("KNN",
                                        tune_KNN=True,
                                        n_classes=self.data[
                                            'info']['n_classes']
                                        )

        tune_duq = TuneDUQModel(self.model,
                                n_classes=self.data['info']['n_classes'],
                                OOD_label=self.OOD_label,
                                data=self.data,
                                device=self.device)
        study = tune_duq.run_optuna()
        return k, study

    def _tune_auroc_threshold(self) -> dict:
        """
        Determines if the AUROC on a given ID + OOD set
        is below 0.5 per method.

        Returns:
            dict: Contains all tested methods as a key
                which, when indexed, return a bool determining
                whether to invert the UQ score on another data split.
        """
        data2 = self.data.copy()
        data2['test'] = data2.get('val_2')
        self.OOD_pipeline.scores = {}
        self.OOD_pipeline.set_data_label(
            data=data2, OOD_label=self.OOD_label_2)

        invert = {}

        y_true = self.OOD_pipeline.ground_truth_labels()

        for method in self.methods:
            auroc = self._get_auroc_raw(
                y_true, method)
            if auroc < 0.5:
                invert[method] = True
                self.inversed[method] += 1
            else:
                invert[method] = False

        return invert

    def _tune(self,
              inverse: bool,
              subject: int
              ) -> dict:
        """
        A pipeline for the hyperparameter tuning per ID OOD combination.
        Saves the found hyperparameters to a file when tuning finishes.

        Args:
            inverse (bool): Whether to tune for the inverse experiment or not.
            subject (int): subject for whom to tune the hyperparameters.

        Returns:
            dict: Contains the found hyperparameters.
                Indexed by OOD class label.
        """
        print("Tuning started...")
        label_k1 = self.len_classes - 1
        label_k2 = self.len_classes - 2

        if inverse:
            values = defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(float)))

            for key1, key2 in combinations(self.class_keys, 2):
                self._get_data_and_model(
                    subject=subject,
                    OOD_class_key=[key1, key2]
                )

                def run_tune(store_key_a, store_key_b):
                    k, study = self._OOD_parameter_tuning()
                    values[store_key_a][store_key_b]["k"] = k
                    values[store_key_a][store_key_b].update(study.best_params)

                self._run_inverse_direction(
                    label_k1, label_k2, 'test_k1', 'val_k2',
                    lambda: run_tune(key1, key2)
                )

                self._run_inverse_direction(
                    label_k2, label_k1, 'test_k2', 'val_k1',
                    lambda: run_tune(key2, key1)
                )

        else:
            values = defaultdict(dict)
            for key in self.class_keys:
                self._get_data_and_model(
                    subject=subject,
                    OOD_class_key=key)
                k, study = self._OOD_parameter_tuning()
                values[key]["k"] = k
                values[key].update(study.best_params)

        with open(f"Hyperparam_{self.dataset_name}.json", "w") as f:
            json.dump(values, f, indent=2)

        print("Tuning finished")
        return values

    def _get_task_auroc(self) -> None:
        """
        Gets the on-task (macro) AUROC for a trained model
        given an ID dataset test split.
        """
        n_classes = self.data['info']['n_classes']
        use_duq = 'DUQ' in self.methods

        self.model.eval()
        outputs = {'main': [], 'duq': []}
        labels = []

        with torch.no_grad():
            for data, label in self.data['test_ID']:
                data = data.to(self.device)
                labels.append(label.to(self.device))

                logits = self.model(data)
                outputs['main'].append(F.softmax(logits, dim=1))

                if use_duq:
                    outputs['duq'].append(self.OOD_pipeline.duq.model(data))

        labels = torch.cat(labels).cpu().numpy()
        outputs = {
            k: torch.cat(v).cpu().numpy()
            for k, v in outputs.items() if v
        }

        def compute_auroc(y_true: np.ndarray,
                          y_score: np.ndarray
                          ) -> float:
            """
            Computes (macro) AURCO scores.

            Args:
                y_true (np.ndarray): Ground truth labels.
                y_score (np.ndarray): Predictions.

            Returns:
                float: AUROC score
            """
            if n_classes == 2:
                return roc_auc_score(y_true, y_score[:, 1])
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
            return roc_auc_score(
                y_true_bin, y_score, average='macro', multi_class='ovr'
            )

        self.val_auroc.append(compute_auroc(labels, outputs['main']))

        if use_duq:
            self.duq_auroc.append(compute_auroc(labels, outputs['duq']))

    def _get_auroc_raw(self,
                       y_true: np.array,
                       method: str,
                       MC_simulations: int = 50,
                       deep_n: int = 5,
                       invert: bool = False
                       ) -> float:
        """
        Computes the AUROC score for a test set containing ID + OOD data
        on the uncertainty scores provided by the requested method.

        Args:
            y_true (np.array): The binary ground truth labels (ID vs OOD).
            method (str): The UQ method.
            MC_simulations (int, optional): Number of simulations for
                MC Dropout. Defaults to 50.
            deep_n (int, optional): Number of ensemble memnbers. Defaults to 5.
            invert (bool, optional): Whether to invert the uncertainty scores.
                Defaults to False.

        Returns:
            float: AUROC of the UQ method on OOD detection.
        """

        score = self.OOD_pipeline.get_score(
            method, MC_t=MC_simulations, deep_n=deep_n)

        score = np.array(score)
        if invert:
            score *= -1
        return roc_auc_score(y_true, score)

    def _score_for_all(self,
                       invert: None | dict = None,
                       ReAct: bool = False
                       ) -> None:
        """
        A pipeline to evaluate the OOD detection performance of
        all the methods specified as a class attribute for all different
        variations (e.g. LOCO, with ReAct and with Inversion).

        Args:
            invert (None | dict, optional):
                If none the normal LOCO experiment is executed. Else,
                it takes into account inversion of the scores.
                Defaults to None.
            ReAct (bool, optional): Whether to evaluate the methods with the
                ReAct supplement. Defaults to False.
        """

        self.OOD_pipeline.scores = {}
        self.OOD_pipeline.set_data_label(
            data=self.data, OOD_label=self.OOD_label)

        def _get_scores(save_result: any,
                        inv: bool = False
                        ) -> None:
            """
            Loops over all methods to evaluate their OOD detection
            perfromance.

            Args:
                save_result (any): Where to save results (class attribute).
                inv (bool, optional): Whether to take into account inversion.
                    Defaults to False.
            """

            y_true = self.OOD_pipeline.ground_truth_labels()
            inversion = False

            for method in self.methods:
                if inv:
                    inversion = invert.get(method, False)
                    if not inversion:
                        continue
                auroc = self._get_auroc_raw(
                    y_true, method, deep_n=5, invert=inversion)
                save_result[method].append(auroc)

        _get_scores(self.results)

        if "Deep Ensemble" in self.methods:
            self.all_ens_auroc.append(self.OOD_pipeline.get_ensemble_auroc())

        if invert is not None:
            for method in self.methods:
                if not invert.get(method, False):
                    self.inversion[method].append(self.results[method][-1])

            _get_scores(self.inversion, inv=True)

        if ReAct:
            self.OOD_pipeline.activate_ReAct()
            n, m, std, diff_1 = self.OOD_pipeline.get_activation_difference()
            self.act_diff['N'].append(n)
            self.act_diff['M'].append(m)
            self.act_diff["std"].append(std)
            self.act_diff["Mean Max Activation difference"].append(diff_1)
            _get_scores(self.results_ReAct)
            self.OOD_pipeline.deactivate_ReAct()

    def _get_data(self,
                  subject: int,
                  OOD_class_key: list | str,
                  discard: None | str = None
                  ) -> None:
        """
        Gets the data from the dataloader and stores it in
        a dict specified as the class attribute 'self.data'.
        The number of datasplits returned by the dataloader depends
        on the number of specified OOD classes.

        Args:
            subject (int): The subject for whom to load the data.
            OOD_class_key (list | str): The class(es) marked as OOD.
            discard (None | str, optional): A class that is to be discarted
                from the experiment. Defaults to None.
        """

        if discard is not None:
            self.OOD_label -= 1

        self.data = (
            self.data_loader.load_data_subject(
                subject=subject, OOD_class=OOD_class_key, discard=discard)
        )

    def _get_model(self) -> None:
        """Initialize and train an EEGNeX model with the loaded data.
        The model is saved as a class attribute.
        """
        info = self.data['info']
        model = EEGNeX_8_32(info['n_classes'],
                            info['n_timesteps'],
                            info['n_channels'],
                            drop_prob=0.5).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = Trainer(model, optimizer, self.criterion, self.device)
        self.model = trainer.train(self.data, n_epochs=200)

    def _get_data_and_model(self,
                            subject: int,
                            OOD_class_key: list | str,
                            discard: str = None,
                            model: bool = True
                            ) -> None:
        """
        Calls functions to load the data and the model.

        Args:
            subject (int): Subject for whom to load the data.
            OOD_class_key (list | str):The class(es) marked as OOD.
            discard (str, optional): A class that is to be discarted
                from the experiment. Defaults to None.
            model (bool, optional): Whether to initialize and train a model.
                Defaults to True.
        """

        self._get_data(subject=subject,
                       OOD_class_key=OOD_class_key,
                       discard=discard)
        if model:
            self._get_model()

    def _save_results(self,
                      subject: int
                      ) -> None:
        """
        Saves the AUROC scores to a file when the experiment is completed.

        Args:
            subject (int): Subject for whom the experiment was executed.
        """
        if subject is None:
            subject = 'all'

        with open(f"results_{self.dataset_name}_{subject}.json", "w") as f:
            json.dump({
                "Subjects": (len(self.subject_list)-1),
                "results": self.results,
                "val_auroc": self.val_auroc,
                "deep_ensamble_auroc": self.all_ens_auroc,
                "duq_val_auroc": self.duq_auroc
            }, f, indent=2)

        if any(self.results_ReAct.values()):
            with open(
                    f"results_ReAct_{self.dataset_name}_{subject}.json",
                    "w") as f:
                json.dump({
                    "Subjects": (len(self.subject_list)-1),
                    "differences": self.act_diff,
                    "results": self.results_ReAct,
                }, f, indent=2)

        if any(self.inversion.values()):
            with open(
                    f"results_inversion_{self.dataset_name}_{subject}.json",
                    "w") as f:
                json.dump({
                    "results": self.inversion,
                    "count": self.inversed
                }, f, indent=2)

    def _data_methods(self,
                      subject: int,
                      OOD_class: str | list,
                      param: dict,
                      ) -> None:
        """
        Call a function to load the data, model, intialize the OOD pipeline
        and get the on-task performance of the trained model.

        Args:
            subject (int): The subject for whom to execute the experiment.
            OOD_class (str| list): The name(s) of the class(es) marked as OOD.
            param (dict): Hyperparameters for KNN and DUQ for chosen OOD class.
        """

        self._get_data_and_model(subject, OOD_class)
        self.OOD_pipeline = OOD(model=self.model,
                                device=self.device,
                                data=self.data,
                                OOD_label=self.OOD_label,
                                methods=self.methods,
                                param=param)
        self._get_task_auroc()

    def _normal_results(self,
                        subjects: list[int],
                        values: dict,
                        ReAct: bool = False
                        ) -> None:
        """
        Run the experiment for the basic LOCO experiment.

        Args:
            subjects (list[int]): List of subjects.
            values (dict): Hyperparameters for KNN and DUQ for all OOD classes.
            ReAct (bool, optional): Whether to run the experiment with react.
                Defaults to False.
        """

        for subject in subjects:
            for key in self.class_keys:
                self._data_methods(
                    subject=subject, OOD_class=key, param=values[key]),
                self._score_for_all(ReAct=ReAct)

    def _run_inverse_direction(self,
                               label_main: int,
                               label_aux: int,
                               test_key: str,
                               val_key: str,
                               fn: any,
                               param: None | dict = None,
                               ) -> any:
        """
        A helper function to switch around validation and test splits.

        Args:
            label_main (int): Main OOD label
            label_aux (int): Secondary OOD label
            test_key (str): Key of the proposed test set.
            val_key (str): Key of the proposed OOD validation set.
            fn (any): Function call.

        Returns:
            any: An executed function.
        """

        self.data['test'] = self.data.get(test_key)
        self.data['val_2'] = self.data.get(val_key)

        self.OOD_label = label_main
        self.OOD_label_2 = label_aux

        if param is not None:
            self.OOD_pipeline = OOD(model=self.model,
                                    device=self.device,
                                    data=self.data,
                                    OOD_label=self.OOD_label,
                                    methods=self.methods,
                                    param=param)
        return fn()

    def _inverse_score_results(self,
                               subjects: list[int],
                               values: dict,
                               ReAct: bool = False
                               ) -> None:
        """
        Run the LOCO experiment with inversion. This uses 2 ID classes and
        2 OOD classes, where one is used in order to asses whether to use
        inversion which is tested on the other.

        Args:
            subjects (list[int]): List of subjects.
            values (dict): Hyperparameters for KNN and DUQ for all OOD classes.
            ReAct (bool, optional): Whether to run the experiment with react.
                Defaults to False.
        """
        order = []
        label_k1 = self.len_classes - 1
        label_k2 = self.len_classes - 2

        for subject in subjects:
            for key1, key2 in combinations(self.class_keys, 2):
                order.append((key1, key2))
                self._data_methods(
                    subject=subject,
                    OOD_class=[key1, key2],
                    param=values[key1][key2],
                )

                def score():
                    invert = self._tune_auroc_threshold()
                    self._score_for_all(invert=invert, ReAct=ReAct)

                self._run_inverse_direction(
                    label_k1, label_k2, 'test_k1', 'val_k2', score
                )

                self._run_inverse_direction(
                    label_k2, label_k1, 'test_k2', 'val_k1', score,
                    param=values[key2][key1]
                )

        with open(f"inversed_{self.dataset_name}_order.json", "w") as f:
            json.dump(order, f, indent=2)

    def experiment(self,
                   tune: bool = False,
                   inverse: bool = False,
                   subject: int | None = None,
                   ReAct: bool = False
                   ) -> None:
        """
        Main function call that loads the hyperparameters and decides
        which experiment will be executed.

        Args:
            tune (bool, optional): Whether to perform hyperparameter tuning
                for DUQ or KNN. Defaults to False.
            inverse (bool, optional): Whether to perform the inversion
                experiment. Defaults to False.
            subject (int | None, optional): The subjects for whom to run the
                experiment. Defaults to None.
            ReAct (bool, optional): Wether to use ReAct in the experiment.
                Defaults to False.
        """

        self._reset_init()

        if tune:
            values = self._tune(inverse=inverse, subject=subject)
            print(values)
        else:
            if inverse:
                name = self.dataset_name + "_inverse"
            else:
                name = self.dataset_name
            param = hyperparam().get_hyperparameters(name)

        if not tune:
            if subject is not None:
                subjects = [self.subject_list[subject-1]]
            else: 
                subjects = self.subject_list[1:]

            if inverse:
                print("Inversion in experiment activated")
                self._inverse_score_results(
                    subjects, values=param, ReAct=ReAct)
                inverse = 'inverse'
            else:
                self._normal_results(
                    subjects, values=param, ReAct=ReAct)
                inverse = 'normal'

            self._save_results(subject)
