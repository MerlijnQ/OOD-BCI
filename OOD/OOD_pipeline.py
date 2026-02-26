import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict

from OOD.ReAct import ReAct
from OOD.deep_ensemble import DeepEnsemble
from OOD.KNN import KNN
from OOD.DDU import DDU
from OOD.DUQ import DUQ
from OOD.energy import energy_score
from OOD.softmax import softmax_ood_score
from OOD.MC_dropout import MC_ood_score
from torch.utils.data import DataLoader


class OOD:
    """
    Pipeline for performing out-of-distribution (OOD) detection using
    multiple uncertainty quantification (UQ) methods.

    This class manages:
        - Inference on a trained model.
        - Initialization and tracking of various OOD detection methods
          including Softmax, Energy , KNN, DUQ, DDU, MC Dropout, Deep Ensemble.
        - Integration with ReAct for rectified activations.

    Attributes:
        model (nn.Module): Trained model for inference. Should support a
            `feature_extractor()` method for some methods.
        device (Union[str, torch.device]): Device for model inference
            ('cpu' or 'cuda').
        data (Dict[DataLoader]): Dictionary containing
            'train', 'val', and 'test' splits.
        OOD_label (int): Label in the dataset to consider as OOD.
        methods (List[str]): List of OOD methods to benchmark.
        param (Dict): Hyperparameters for DUQ and KNN methods.
        use_ReAct (bool): Flag indicating whether ReAct is applied.
        r (ReAct | None): ReAct class applying threshold.
        threshold (torch.Tensor | None): Threshold for OOD decision.
        _scores (Dict): Dictionary storing uncertainty/OOD scores for multiple
            methods that can be computed without multiple forward passes.
        _knn, _ddu, _duq, _de: Placeholders for corresponding method models.
    """
    def __init__(self,
                 model: nn.Module,
                 device: torch.cuda.device | str,
                 data: dict[DataLoader] = None,
                 OOD_label: int = None,
                 methods: list = [],
                 param: dict = {}
                 ) -> None:
        """
        Initialize the OOD detection pipeline.

        Args:
            model (nn.Module): A trained model on which inference can be
                performed. Some methods require the model to have a
                `feature_extractor()` function.
            device (Union[str, torch.device]): Device for model inference
                ('cpu' or 'cuda').
            data (Optional[Dict[str, DataLoader]], optional): Dictionary
                containing 'train', 'val', and 'test' DataLoaders.
                Defaults to None.
            OOD_label (int): Label of the OOD classs. Defaults to None.
            methods (Optional[List[str]], optional): List of OOD methods to
                benchmark. Defaults to empty list.
            param (Optional[Dict], optional): Hyperparameters for DUQ and KNN.
                Defaults to empty dict.

        Raises:
            ValueError: If `OOD_label` is not specified.
        """
        if OOD_label is None:
            raise ValueError("Indicate which label in the dataset is selected \
                              as OOD")

        self.model = model
        self.device = device

        self.data = data
        self.OOD_label = OOD_label

        self.methods = methods
        self.param = param

        # Intialize attributes used for ReAct
        self.use_ReAct = False
        self.r = None
        self.threshold = None

        # Initialize attributes to keep track of the uncertainty scores.
        self._scores = {}

        # Initialize attributes to which corresponding models belonging
        # to these methods will be assigned.
        self._knn = None
        self._ddu = None
        self._duq = None
        self._de = None

        if self.data is None:
            print("Dont forget to set data and label")
        if not self.param:
            print("No hyperparameters provided to pipeline.\
                   Cannot initialize KNN and DUQ models")
            self.set_ID_vectors(
                self.data['train'], KNN_score=False, DUQ_score=False)
        else:
            self.set_ID_vectors(self.data['train'])

    def set_data_label(self,
                       data: dict[DataLoader],
                       OOD_label: int,
                       reinitialize: bool = False
                       ) -> None:
        """
        Change the data and OOD label attributes.

        Args:
            data (dict[DataLoader]): A dictionary containing the different
                datasplits. The keys should contain 'train' and 'val' for the
                corresponding splits. The 'test' key should be present for
                calculating the entropy on ID/OOD samples.
            OOD_label (int): The label of the OOD class in the dataset.
            reinitialize (bool, optional): Whether to reinitialize the ID
                vectors and ReAct threshold if the data and OOD label are changed.
                Defaults to False.

        """
        self.OOD_label = OOD_label
        self.data = data
        self.scores.clear()
        if reinitialize:
            self.set_ID_vectors(ID_loader=self.data['train'])
            if self.use_ReAct:
                self.threshold = self.r.get_clamp(self.data['val'])

    def activate_ReAct(self) -> None:
        """
        Activates rectified activations in the experiment.
        This clears any scores computed previously.

        Raises:
            Warning: Triggers is ReAct was already activated.
        """
        if self.use_ReAct:
            raise Warning(
                "Tried to activate ReAct when it was already activated")

        self._scores.clear()

        print("Activating ReAct...")
        self.r = ReAct(device=self.device)
        self.use_ReAct = True
        self.threshold = self.r.get_clamp(self.data['val'],
                                          model=self.model)
        self.set_ID_vectors(ID_loader=self.data['train'])

    def deactivate_ReAct(self) -> None:
        """
        Deactivates rectified activations in the experiment if it was
        active. This clears any scores and thresholds computed previously.

        Raises:
            Warning: Triggers when ReAct was never activated in the first
                place.
        """
        if not self.use_ReAct:
            raise Warning(
                "Tried to deactivate ReAct when it was never activated")

        print("Deactivating ReAct...")
        self._scores.clear()
        self.r = None
        self.use_ReAct = False
        self.threshold = None
        self.set_ID_vectors(ID_loader=self.data['train'])

    def get_activation_difference(self) -> tuple[float]:
        """
        Calculates and the difference in activations.

        Returns:
            tuple[float]: The n times activations for an OOD samples were
                higher, mean difference, standard deviation, mean max
                difference.
        """
        return self.r.get_difference_activations(
            self.data['test'], self.OOD_label, model=self.model)

    def get_ensemble_auroc(self) -> float:
        """
        Gets the on task (macro) auroc of the trained ensemble.

        Raises:
            ValueError: Triggers when the ensemble has not been trained yet.

        Returns:
            float: The on-task ensemble auroc.
        """
        if self._de is None:
            raise ValueError("Ensemble AUROC not computed yet. \
                              Please run deep_ensemble_score first.")
        return self._de.mean_auroc

    def _get_ID_z(self,
                  ID_loader: DataLoader
                  ) -> tuple[torch.Tensor, np.ndarray]:
        """
        Calculates all the ID feature vectors (optionally with ReAct when
        activated).

        Args:
            ID_loader (DataLoader): A PyTorch Dataloader object with (raw)
                ID data.

        Returns:
            tuple[torch.Tensor, np.ndarray]: The feature vectors and their
                correspondng labels
        """

        self.model.eval()
        self.model.to(self.device)

        with torch.no_grad():
            z_list = []
            label_list = []
            for data, labels in ID_loader:
                data = data.to(self.device)
                z = self.model.feature_extractor(data)
                if self.use_ReAct:
                    z = self.r.set_clamp(z, self.threshold)
                z_list.append(z.detach().cpu())
                label_list.append(labels.detach().cpu().numpy())
            z_list = torch.cat(z_list, dim=0)
            label_list = np.concatenate(label_list, axis=0)
            return z_list, np.array(label_list)

    def set_ID_vectors(self,
                       ID_loader: DataLoader,
                       DDU_score: bool = True,
                       KNN_score: bool = True,
                       DUQ_score: bool = True
                       ) -> None:
        """
        Calculates the ID feature vectors and fits the UQ models
        that rely on these vectors if in the list with methods and true.
        Removes the ID vectors afterwards to save memory.

        Args:
            ID_loader (DataLoader): A PyTorch Dataloader object with (raw)
                ID data.
            DDU_score (bool, optional): Fits the DDU model. Defaults to True.
            KNN_score (bool, optional): Fits the KNN model. Defaults to True.
            DUQ_score (bool, optional): Fits the DUQ model. Defaults to True.
        """

        z, ID_labels = self._get_ID_z(ID_loader)
        n_classes = len(np.unique(ID_labels))

        if KNN_score and "KNN" in self.methods:
            max_k = int(len(ID_labels)/n_classes)
            self._knn = KNN(ID_features=z, max_k=max_k)

        if DDU_score and 'DDU' in self.methods:
            self._ddu = DDU(ID_features=z,
                            device=self.device,
                            n_classes=n_classes,
                            ID_labels=ID_labels)

        if DUQ_score and 'DUQ' in self.methods:
            gamma = 1 - self.param['one_minus_gamma']
            self._duq = DUQ(feature_extractor=self.model,
                            num_classes=n_classes,
                            optimizer=optim.Adam,
                            centroid_size=self.param['centroid_size'],
                            gamma=gamma)
            self._duq.train(self.data['train'],
                            self.data['val'],
                            penalty_w=self.param['penalty_w'],
                            threshold=self.threshold)

        del z

    def ground_truth_labels(self) -> list:
        """
        Determines the ground truth labels for OOD detection.
        1 Indicates and OOD label and 0 and ID label.

        Returns:
            list: A list with binary label to indicate if a sample is OOD.
        """
        y_true = []

        with torch.no_grad():
            for _, labels in self.data['test']:
                if isinstance(labels, torch.Tensor):
                    labels = labels.to(self.device)
                    labels = labels.detach().cpu().numpy()

                values = (labels == self.OOD_label).astype(int)
                y_true.extend(values)

            return y_true

    def _get_z_dependent_scores(self,
                                ) -> None:
        """
        Calculates the uncertainty scores for methods that do not depend
        on multiple forward passes or different model initializations.
        This includes softmax, energy, DUQ, DDU and KNN. The scores are
        only calculated if they are part of the methods list provided as a
        class attribute. The computed scores are stored as a class attribute
        and will be returned by the get_score() function.
        """

        score_fns = {
            "Softmax": lambda _, logits: softmax_ood_score(logits),
            "Energy": lambda _, logits: energy_score(logits),
            "DUQ": lambda feats, _: self._duq.DUQ_ood_score(feats),
            "DDU": lambda feats, _: self._ddu.DDU_ood_score(
                z_test=feats),
            "KNN": lambda feats, _: self._knn.KNN_ood_score(
                z_test=feats,
                k=self.param['k']
                )
        }
        logit_methods = {"Softmax", "Energy"}

        scores = defaultdict(list)
        if self.model.training:
            self.model.eval()

        with torch.no_grad():
            for data, _ in self.data["test"]:
                data = data.to(self.device)

                features = self.model.feature_extractor(data)

                if self.use_ReAct:
                    features = self.r.set_clamp(features, self.threshold)

                classification_logits = None
                if logit_methods & set(self.methods):
                    classification_logits = self.model.classify(features)

                valid_methods = set(self.methods) & set(score_fns.keys())
                for method in valid_methods:
                    scores[method].extend(
                        score_fns[method](features, classification_logits)
                    )

            del features, classification_logits

        self._scores = scores

    def _de_score(self,
                  deep_n: int = 5,
                  criterion:
                  torch.nn.modules.loss._Loss = nn.CrossEntropyLoss(),
                  **kwargs: dict
                  ) -> np.ndarray:
        """
        If an ensemble is not trained yet it will train an ensemble with n
        members and calculate the predictive entropy afterwards on the test
        set containing OOD data.

        Args:
            deep_n (int, optional): The number of ensemble members.
                Defaults to 5.
            criterion (torch.nn.modules.loss._Loss, optional): The criterion
                used for model training. Defaults to nn.CrossEntropyLoss().

        Returns:
            np.ndarray: The predictive entropy for the samples in the test set.
        """

        if self._de is None:

            first_model = self.model.state_dict(

            ) if self.model is not None else None

            self._de = DeepEnsemble(data=self.data,
                                    first_model=first_model,
                                    criterion=criterion,
                                    n=deep_n)

        return self._de.ensemble_ood_score(use_ReAct=self.use_ReAct,
                                          r=self.r,
                                          dataloader=self.data['test'])

    def get_score(self,
                  method: str,
                  tune_KNN: bool = False,
                  **kwargs) -> np.ndarray:
        """
        Get the certainty scores for any of the implemented methods.

        Args:
            method (str): The methods to use.
            tune_KNN (bool, optional): Whether to tune the k hyperparameter
                value. Defaults to False.

        Raises:
            ValueError: Raised when the method requested is is not implemented.

        Returns:
            np.ndarray: The uncertainty scores for the requested method.
        """
        if tune_KNN and method == "KNN":
            if self._knn is None:
                self.set_ID_vectors(ID_loader=self.data.get('train'),
                                    DDU_score=False,
                                    KNN_score=True,
                                    DUQ_score=False)

            return self._knn.tune_k(model=self.model,
                                    test_loader=self.data['test'],
                                    OOD_label=self.OOD_label,
                                    device=self.device,
                                    use_ReAct=self.use_ReAct,
                                    r=self.r,
                                    threshold=self.threshold)

        if method in ["Softmax", "Energy", "DDU", "KNN", "DUQ"]:
            if not self._scores:
                self._get_z_dependent_scores()
            return self._scores[method]

        match method:
            case "MC Dropout":
                return MC_ood_score(test_loader=self.data['test'],
                                    model=self.model,
                                    threshold=self.threshold,
                                    r=self.r,
                                    device=self.device,
                                    **kwargs)
            case "ReAct":
                return self.get_activation_difference()
            case "Deep Ensemble":
                return self._de_score(**kwargs)
            case _:
                raise ValueError("Method not found")
