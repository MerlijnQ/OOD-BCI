from moabb.paradigms import MotorImagery
import numpy as np
import os
import moabb.datasets as md
import inspect
import moabb
from moabb.datasets.base import BaseDataset
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from utils.dataProcessor import Processor
import mne
from torch.utils.data import DataLoader


class PickChannels(BaseEstimator, TransformerMixin):
    """
    Postprocessor to select a subset of EEG channels.

    Useful for preprocessing EEG datasets to include only specific channels
    for model training or analysis when channels differ per session.

    Attributes:
        channels (List[str]): List of channel names to select from the dataset.
    """
    def __init__(self,
                 channels: list[str],
                 ) -> None:
        """
        Initializes the channel selector

        Args:
            channels (list[str]): Names of the channels to select
        """
        super().__init__()
        self.channels = channels

    def fit(self, X, y=None):
        """Method to comply with APIs.
        """
        return self

    def transform(self,
                  X: mne.Epochs | mne.io.Raw
                  ) -> mne.Epochs | mne.io.Raw:
        """
        Select the specified channels from the input data.

        Args:
            X (mne.Epochs | mne.io.Raw):
                Input with all channels.

        Returns:
            mne.Epochs | mne.io.Raw:
                Output with only selected channels.
        """

        if hasattr(X, "pick_channels"):
            X.pick_channels(self.channels)
            return X
        elif isinstance(X, np.ndarray):
            raise ValueError("Numpy input not supported."
                             "Pick channels from raw epochs.")


class dataLoader():
    """
    Custom MOABB data loader for preparing EEG datasets for training and
    evaluation.

    Handles train/validation/test splits, preprocessing, and dataset caching.

    Attributes:
        dataset_name (str): Name of the MOABB dataset to load.
        paradigm (Paradigm): MOABB paradigm object specifying the type of data.
        batch_size (int): Batch size used for training and validation
            DataLoaders.
        test_batch_size (int): Batch size used for test DataLoader.
        test_size (float): Fraction of the dataset reserved for testing.
        validation_size (float): Fraction of the training set reserved for
            validation.
        _dataset: Loaded MOABB dataset object.
        cache_base (str): Base path for dataset cache.
        _processor (Processor): Processor object handling splits and batch
            preparation.
    """
    def __init__(self,
                 dataset: str,
                 test_size: float = 0.15,
                 validation_size: float = 0.1,
                 test_batch_size: None | int = None,
                 batch_size: int = 32,
                 paradigm: moabb.paradigms = MotorImagery()
                 ) -> None:
        """
        Initialize the MOABB dataLoader.

        Args:
            dataset (str): Name of the MOABB dataset to load.
            test_size (float, optional): Fraction of the dataset to use as
                test set. Defaults to 0.15.
            validation_size (float, optional): Fraction of training set to use
                as validation. Defaults to 0.1.
            test_batch_size (Optional[int], optional): Batch size for test
                DataLoader. Defaults to None (uses batch_size).
            batch_size (int, optional): Batch size for training/validation
                DataLoaders. Defaults to 32.
            paradigm (Paradigm, optional): MOABB paradigm specifying the type
                of data. Defaults to MotorImagery().

        Raises:
            ValueError: If test_size or validation_size is not between 0 and 1.
        """
        if test_size < 0 or test_size > 1:
            raise ValueError("Test size should be between 0 and 1")

        if validation_size < 0 or validation_size > 1:
            raise ValueError("Validation size should be between 0 and 1")

        self.paradigm = paradigm
        self.batch_size = batch_size
        self.test_size = test_size
        self.validation_size = validation_size
        self.dataset_name = dataset

        if test_batch_size is None:
            self.test_batch_size = batch_size
        else:
            self.test_batch_size = test_batch_size

        self._dataset = self._load_dataset()
        self.cache_base = "/cache"

        self._processor = Processor(test_size=self.test_size,
                                    validation_size=self.validation_size,
                                    batch_size=self.batch_size,
                                    test_batch_size=self.test_batch_size)

    def _load_dataset(self) -> BaseDataset:
        """
        Initializes the moabb dataset object.

        Raises:
            ValueError: Raised if dataset does not exist in MOABB.

        Returns:
            BaseDataset: The MOABB dataset instance.
        """
        WHITELIST = {
            "bnci2014_001": "BNCI2014_001",
        }
        name_map = {
            name.lower(): name
            for name in dir(md)
        }

        key = self.dataset_name.lower()
        if key not in name_map:
            raise ValueError(f"Dataset '{key}' not found")

        if key in WHITELIST:
            cls_name = WHITELIST[key]
            dataset_cls = getattr(md, cls_name)
            return dataset_cls()

        dataset_cls = None
        dataset_module = getattr(md, name_map[key])
        for _, obj in inspect.getmembers(dataset_module):
            if (
                inspect.isclass(obj) 
                and issubclass(obj, BaseDataset) 
                and not inspect.isabstract(obj)
            ):
                dataset_cls = obj
                break

        if dataset_cls is None:
            raise ValueError(
                f"No concrete dataset class found in module '{name_map[key]}'")

        return dataset_cls()

    def return_subjects(self) -> list:
        """
        Returns:
            list: A list of subjects in the dataset
        """
        return self._dataset.subject_list

    def get_classes(self) -> list:
        """
        Returns:
            list: The classes present in the dataset.
        """
        return self._dataset.event_id

    def _enforce_channel_consistency(self,
                                     subject: int
                                     ) -> tuple[list[str], list[str]]:
        """
        This function enforces the same channels for all samples
        in the part of a dataset belonging to a specific subject to
        prevent shape mismatches.

        Args:
            subject (int): The subjects for whom the data is loaded.

        Returns:
            tuple[list[str], list[str]]: The common channels for all processed
                samples and all found channels.
        """

        def _get_valid_channels(raw: mne.io.BaseRaw
                                ) -> list[str]:
            """
            Picks the EEG channels in a raw sample.

            Args:
                raw (mne.io.BaseRaw): A raw mne data object.

            Returns:
                list[str]: The list of valid channel names for a specific
                    sample.
            """
            raw.pick_types(eeg=True)
            raw_ch_names = raw.info["ch_names"]
            return raw_ch_names

        raws = []
        all_channels = []

        X, _, _ = self.paradigm.get_data(
                        self._dataset,
                        subjects=[subject],
                        return_raws=True
                    )

        for x in X:
            if isinstance(x, list):
                raws.extend(x)
            else:
                raws.append(x)

        all_channels = []
        channels = []
        for raw in raws:
            ch_names = _get_valid_channels(raw)
            channels.append(set(ch_names))
            all_channels.append(ch_names)

        common_channels = sorted(set.intersection(*channels))
        print(f"Common channels across all runs: {len(common_channels)}")
        del X, raws
        return common_channels, all_channels

    def _load_data(self,
                   subject: int
                   ) -> tuple[np.ndarray, np.ndarray]:
        """
        Load the data for a specified subject.
        The selection of channels specified for Schirrmeister come from
        https://github.com/robintibor/high-gamma-dataset/blob/master/example.py

        Processing is done using the MOABB pipeline according to the specified
        paradigm. The default paradigm is Motor Imagery.

        Args:
            subject (int): The subject for whom to load the data.

        Returns:
            tuple[np.ndarray, np.ndarray]: The samples and their labels.
        """

        if self.dataset_name == "Stieger2021":
            common_channels, _ = (
                self._enforce_channel_consistency(subject)
                )
            postprocess_pipeline = Pipeline([
                ("pick", PickChannels(common_channels))
            ])
        if self.dataset_name == "Schirrmeister2017":
            common_channels = [
                'FC5', 'FC1', 'FC2', 'FC6', 'C3', 'C4', 'CP5',
                'CP1', 'CP2', 'CP6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2',
                'C6', 'CP3', 'CPz', 'CP4', 'FFC5h', 'FFC3h', 'FFC4h',
                'FFC6h', 'FCC5h', 'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h',
                'CCP3h', 'CCP4h', 'CCP6h', 'CPP5h', 'CPP3h', 'CPP4h',
                'CPP6h', 'FFC1h', 'FFC2h', 'FCC1h', 'FCC2h',
                'CCP1h', 'CCP2h', 'CPP1h', 'CPP2h',
            ]
            postprocess_pipeline = Pipeline([
                ("pick", PickChannels(common_channels))
            ])
        else:
            postprocess_pipeline = None

        X, y, _ = self.paradigm.get_data(
            self._dataset, subjects=[subject],
            postprocess_pipeline=postprocess_pipeline
            )

        print("Number of channels: ", X.shape[1])

        return X, y

    def _cache_original(self,
                        subject: int,
                        X: np.ndarray,
                        y: np.ndarray
                        ) -> None:
        """
        Caches the loaded data to an mmap file. This prevents having to
        reload the dataset using the MOABB pipeline when changing OOD label.
        The cached file for the previous subject is automatically deleted if
        detected.

        Args:
            subject (int): The subject for whom to load the data.
            X (np.ndarray): The samples belonging to the specified subject.
            y (np.ndarray): The labels belonging to the loaded samples.

        Raises:
            e: Failed to delete old cached files.
        """
        cache_dir = os.path.join(self.cache_base, self.dataset_name)
        os.makedirs(cache_dir, exist_ok=True)

        try:
            old_subject = subject - 1
            old_X = os.path.join(cache_dir, f"sub{old_subject}_X.dat")
            old_y = os.path.join(cache_dir, f"sub{old_subject}_y.dat")
            old_meta = os.path.join(cache_dir, f"sub{old_subject}_meta.npy")

            for p in [old_X, old_y, old_meta]:
                if os.path.exists(p):
                    os.remove(p)
                    print("Deleted:", p)

        except Exception as e:
            print(f"[cache] Could not delete {p}: {e}")

        X_path = os.path.join(cache_dir, f"sub{subject}_X.dat")
        y_path = os.path.join(cache_dir, f"sub{subject}_y.dat")
        meta_path = os.path.join(cache_dir, f"sub{subject}_meta.npy")

        for p in (X_path, y_path, meta_path):
            if os.path.exists(p):
                os.remove(p)

        try:
            meta = {
                "X_shape": X.shape,
                "X_dtype": X.dtype,
                "y_shape": y.shape,
                "y_dtype": y.dtype,
            }
            np.save(meta_path, meta)

            fp_X = np.memmap(
                X_path,
                dtype=X.dtype,
                mode='w+',
                shape=X.shape
            )
            fp_X[:] = X[:]
            del fp_X  # flush

            fp_y = np.memmap(
                y_path,
                dtype=y.dtype,
                mode='w+',
                shape=y.shape
            )
            fp_y[:] = y[:]
            del fp_y  # flush

            print(f"[cache] Saved subject {subject} → {cache_dir}")

        except Exception as e:
            print(f"[cache] ERROR saving subject {subject}: {e}")

            for p in (X_path, y_path, meta_path):
                if os.path.exists(p):
                    os.remove(p)

            raise e

    def _load_or_cache_subject(self,
                               subject: int
                               ) -> tuple[np.ndarray, np.ndarray]:
        """
        Loads the data from a cached mmap file if available. Else
        the data for a specified subject is loaded using MOABB and cached
        afterwards.

        Args:
            subject (int): The subject for whom to load the data.

        Returns:
            tuple[np.ndarray, np.ndarray]: The samples and their labels.
        """
        cache_dir = os.path.join(self.cache_base, self.dataset_name)
        X_path = os.path.join(cache_dir, f"sub{subject}_X.dat")
        y_path = os.path.join(cache_dir, f"sub{subject}_y.dat")
        meta_path = os.path.join(cache_dir, f"sub{subject}_meta.npy")

        if os.path.exists(X_path):
            meta = np.load(meta_path, allow_pickle=True).item()
            X = np.memmap(
                X_path,
                dtype=meta["X_dtype"],
                mode='r',
                shape=meta["X_shape"]
            )
            y = np.memmap(
                y_path,
                dtype=meta["y_dtype"],
                mode='r',
                shape=meta["y_shape"]
            )

            return X, y

        X, y = self._load_data(subject)
        self._cache_original(subject, X, y)
        return X, y

    def load_data_subject(self,
                          subject: int,
                          OOD_class: str,
                          seed: int = 42,
                          discard: None | str = None
                          ) -> tuple[DataLoader]:
        """
        Load and process the data from MOABB from a dataset for a
        specified subject.

        Args:
            subject (int): Subject for whom to load the data.
            OOD_class (str): The name of the class that will be considered OOD.
            seed (int, optional): Controls randomness. Defaults to 42.
            discard (None | str, optional): The name of a class to discard
                from the data. Defaults to None.

        Returns:
            tuple[DataLoader]: Multiple dataLoader instances depending on the
                number of selected OOD classes.
        """

        X, y = self._load_or_cache_subject(subject)

        X = np.array(X, copy=True)
        y = np.array(y, copy=True)

        if OOD_class is None or isinstance(OOD_class, str):
            return self._processor.load_data_subject_normal(
                X, y, OOD_class=OOD_class, seed=seed, discard=discard)
        elif isinstance(
                OOD_class, list) and all(
                isinstance(i, str) for i in OOD_class):
            return self._processor.load_data_subject_inverse(
                X, y, OOD_class=OOD_class, seed=seed, discard=discard)
        else:
            return ValueError(
                "OOD class should be a string or a list of strings"
                )
