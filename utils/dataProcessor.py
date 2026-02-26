from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder


class EEGDataset(Dataset):
    """
    PyTorch Dataset for EEG data.

    Wraps preprocessed EEG samples and their labels into a Dataset
    compatible with PyTorch DataLoaders.

    Attributes:
        X (torch.Tensor): Tensor of EEG samples with dtype float32.
        y (torch.Tensor): Tensor of corresponding labels with dtype long.
        length (int): Number of samples in the dataset.
    """
    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray
                 ) -> None:
        """
        A Dataset instance used by the dataloaders.

        Args:
            X (np.ndarray): Preprocessed EEG samples
            y (np.ndarray): Labels belonging to the samples
        """
        super().__init__()
        self.X = torch.tensor(X, dtype=torch.float32)  # Convert to tensor
        self.y = torch.tensor(y, dtype=torch.long)     # Labels as long tensor

    def __len__(self) -> int:
        """
        Determines number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.X)  # Total number of trials

    def __getitem__(self,
                    idx: int
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a sample from the dataset and its label based on index.

        Args:
            idx (int): Index of the sample to return.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Sample and its label.
        """
        return self.X[idx].unsqueeze(0), self.y[idx]

    def get_dataset_info(self) -> dict:
        """
        Provides some information about the dataset
        including the number of classes (n_classes),
        the length of the time dimension (n_timesteps) and
        the number of EEG Channels (n_channels)

        Returns:
            dict: Contains the information specified above.
        """
        info = {}
        info['n_classes'] = len(torch.unique(self.y))  # Unique classes labels)
        info['n_timesteps'] = self.X.shape[2]  # Time dimension
        info['n_channels'] = self.X.shape[1]  # EEG channels

        return info


class Processor():
    """
    Processor class to prepare MOABB EEG datasets for model training and
    evaluation.

    Handles train/validation/test splits, batch size configuration for
    PyTorch DataLoaders, z-normalization based on the training split and
    exclusion of OOD classes. 

    Attributes:
        batch_size (int): Batch size used for training and validation
            DataLoaders.
        test_batch_size (int): Batch size used for the test DataLoader.
        test_size (float): Fraction of the dataset reserved for testing.
        validation_size (float): Fraction of the training set reserved fo
            validation.
    """
    def __init__(self,
                 test_size: float = 0.15,
                 validation_size: float = 0.1,
                 test_batch_size: None | int = None,
                 batch_size: int = 32,
                 ):
        """
        Initialize the Processor for MOABB dataset preparation.

        Args:
            test_size (float, optional): Fraction of the dataset to use a
                test set. Defaults to 0.15.
            validation_size (float, optional): Fraction of the training set to
                use as validation set. Defaults to 0.1.
            test_batch_size (Optional[int], optional): Batch size for test
                DataLoader. If None, defaults to `batch_size`.
                Defaults to None.
            batch_size (int, optional): Batch size for training and validation
                DataLoaders. Defaults to 32.

        Raises:
            ValueError: If test_size or validation_size is not between 0 and 1.
        """

        if test_size < 0 or test_size > 1:
            raise ValueError("Test size should be between 0 and 1")

        if validation_size < 0 or validation_size > 1:
            raise ValueError("Validation size should be between 0 and 1")

        self.batch_size = batch_size
        self.test_size = test_size
        self.validation_size = validation_size

        if test_batch_size is None:
            self.test_batch_size = batch_size
        else:
            self.test_batch_size = test_batch_size

    def _return_OOD_split(self,
                          OOD_class: str,
                          class_int: int,
                          X: np.ndarray,
                          y: np.ndarray
                          ) -> tuple[np.ndarray]:
        """
        Splits the data into seperate arrays for
        ID and OOD data.

        Args:
            OOD_class (str): The name of the class marked as OOD.
            class_int (int): The index equivelant of the OOD class label.
            X (np.ndarray): The samples.
            y (np.ndarray): The labels belonging to the samples.

        Raises:
            ValueError: THe OOD class is not a string value.

        Returns:
            tuple[np.ndarray]: ID, label ID, OOD and label OOD arrays.
        """

        if not isinstance(OOD_class, str):
            raise ValueError("OOD class should be a string value")

        idx_OOD = np.where(y == OOD_class)[0]
        idx_ID = np.where(y != OOD_class)[0]

        X_OOD = X[idx_OOD]
        y_OOD = y[idx_OOD]
        X = X[idx_ID]
        y = y[idx_ID]

        y_OOD[:] = class_int
        y_OOD = y_OOD.astype(int)
        print("OOD label and class: ", OOD_class, class_int)
        return X, y, X_OOD, y_OOD

    def _return_loader(self,
                       mean: float,
                       std: float,
                       X: np.ndarray,
                       y: np.ndarray,
                       shuffle: bool = True,
                       data_info: bool = False
                       ) -> DataLoader | tuple[DataLoader, dict]:
        """
        Returns an PyTorch dataloader object for the provided data
        which is added to an EEGDataset object. The data is z-normalised
        around the provided mean and std.

        Args:
            mean (float): The mean used for z-normalization.
            std (float): The standard deviation used for z-normalization.
            X (np.ndarray): The data array.
            y (np.ndarray): Tha labels belonging to the provided data.
            shuffle (bool, optional): Whether to shuffle the data in the
                Dataloader. Defaults to True.
            data_info (bool, optional): Whether to return the information
                dictionary from the created EEGDataset object.
                Defaults to False.

        Returns:
            DataLoader | tuple[DataLoader, dict]: A PyTorch Dataloader with
                optionally a dict containing metadata including
                mean, std, n_classes, n_timesteps and n_channels.
        """

        X = (X - mean) / std

        subject_data = EEGDataset(X, y)
        loader = DataLoader(subject_data,
                            self.test_batch_size,
                            shuffle=shuffle)
        if data_info:
            info = subject_data.get_dataset_info()
            info['mean'] = mean
            info['std'] = std
            return loader, info
        return loader

    def _return_int_labels(self,
                           y: np.ndarray,
                           ) -> np.ndarray:
        """
        Converts the class labels to integer labels.

        Args:
            y (np.ndarray): The class labels per sample in string format.

        Raises:
            ValueError: Raised if labels are not strings.

        Returns:
            np.ndarray: Class labels converted to integer labels.
        """
        if not isinstance(y[0], str):
            raise ValueError("Labels should be strings")

        le = LabelEncoder()
        y = le.fit_transform(y)
        print("ID Class labels: ", le.classes_)
        return y

    def _return_split_concat(self,
                             X: np.ndarray,
                             y: np.ndarray,
                             X_OOD: np.ndarray,
                             y_OOD: np.ndarray
                             ) -> tuple[np.ndarray, np.ndarray]:
        """
        Balances provided ID and OOD data and merges this into one
        dataset.

        Args:
            X (np.ndarray): ID samples.
            y (np.ndarray): Labels belonging to ID samples.
            X_OOD (np.ndarray): OOD samples.
            y_OOD (np.ndarray): Labels belonging to OOD samples.

        Returns:
            tuple[np.ndarray, np.ndarray]: Samples and labels with both ID and
                OOD data.
        """

        ID_size = len(X)
        OOD_size = len(X_OOD)

        # Shuffle ID
        idx = np.random.permutation(ID_size)
        X = X[idx]
        y = y[idx]

        # Shuffle OOD
        idx = np.random.permutation(OOD_size)
        X_OOD = X_OOD[idx]
        y_OOD = y_OOD[idx]

        if ID_size > OOD_size:
            ID_size = OOD_size
            X = X[:ID_size]
            y = y[:ID_size]  
            print("Adjusted ID set size to", OOD_size)
        elif ID_size < OOD_size:
            X_OOD = X_OOD[:ID_size]
            y_OOD = y_OOD[:ID_size]
            print("Adjusted OOD set size to", ID_size)

        n = ID_size * 2
        print("Final test set size (ID + OOD): ", n)

        # #Concatenate ID and OOD data
        X = np.concatenate((X, X_OOD), axis=0)
        y = np.concatenate((y, y_OOD), axis=0)

        # #Shuffle the data
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

        del X_OOD, y_OOD
        return X, y

    def _return_split(self,
                      X: np.ndarray,
                      y: np.ndarray,
                      seed: int = 42,
                      split: int = 2
                      ) -> list[np.ndarray]:
        """
        Splits the data into n splits specified as a percentage
        of the dataset upon class initialization.

        Args:
            X (np.ndarray): Samples.
            y (np.ndarray): Labels of provided samples.
            seed (int, optional): Controls randomness. Defaults to 42.
            split (int, optional): Number of splits. Defaults to 2.

        Returns:
            list[np.ndarray]: Contains multiple arrays belonging to different
                splits in the order [test, labels, val, labels, train, labels].
        """
        splits = []
        size = self.test_size

        for i in range(split):
            X, X_test, y, y_test = train_test_split(X,
                                                    y,
                                                    test_size=size,
                                                    shuffle=True,
                                                    random_state=seed,
                                                    stratify=y)
            size = self.validation_size / (1 - (
                self.test_size + i * self.validation_size)
                )

            splits.extend([X_test, y_test])
        splits.extend([X, y])
        return splits

    def _return_mean_std(self,
                         X: np.ndarray
                         ) -> tuple[float, float]:
        mean = np.mean(X, axis=(0, 2), keepdims=True)
        std = np.std(X, axis=(0, 2), keepdims=True)
        return mean, std

    def _discard(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 discard: str
                 ) -> tuple[np.ndarray, np.ndarray]:
        """
        Discard samples belonging to a specified class.

        Args:
            X (np.ndarray): Samples.
            y (np.ndarray): Labels of provided samples.
            discard (str): Class to discard.

        Returns:
            tuple[np.ndarray, np.ndarray]: Samples and labels without
                discarded class.
        """

        idx = np.where(y != discard)[0]
        X = X[idx]
        y = y[idx]

        return X, y

    def load_data_subject_normal(self,
                                 X: np.ndarray,
                                 y: np.ndarray,
                                 OOD_class: str,
                                 seed: int,
                                 discard: None | str
                                 ) -> dict[DataLoader, DataLoader,
                                           DataLoader, DataLoader, dict]:
        """
        Pipeline to process (label conversion, z-normalisation, OOD-ID
        splitting and concatenation, etc.) the loaded dataset provided that
        only one class is marked as OOD in the Leave-One-Class-Out experiment.

        Args:
            X (np.ndarray): Samples.
            y (np.ndarray): Labels belonging to the provided samples.
            OOD_class (str): The name of the class marked as OOD.
            seed (int): Controls randomness.
            discard (None | str): Optional class to discard.

        Returns:
            dict[DataLoader, DataLoader, DataLoader, DataLoader, dict]:
                The train (ID), validation (ID), test (ID + OOD data),
                ID_test loaders and dataset metadata dict.
        """
        X, y = self._discard(X, y, discard)

        if OOD_class is not None:
            OOD_class_int = int(len(np.unique(np.array(y))) - 1)
            X, y, X_OOD, y_OOD = self._return_OOD_split(OOD_class,
                                                        OOD_class_int,
                                                        X,
                                                        y)

        y = self._return_int_labels(y)

        [X_test, y_test, X_val, y_val, X_train, y_train] = (
            self._return_split(X, y, seed=seed, split=2)
            )

        X_test_ID, y_test_ID = X_test, y_test

        if OOD_class is not None:
            X_test, y_test = (
                self._return_split_concat(X_test, y_test, X_OOD, y_OOD)
            )
            print("Succesfull concatenation of ID test set and OOD data")
        print("Train set size: ", len(X_train),
              "Validation set size: ", len(X_val),
              "Test set size: ", len(X_test))

        mean, std = self._return_mean_std(X_train)
        loader_args = dict(mean=mean, std=std, data_info=False, shuffle=False)

        train_loader, info = self._return_loader(
                                mean=mean,
                                std=std,
                                X=X_train,
                                y=y_train,
                                data_info=True,
                                shuffle=True
                            )

        validation_loader = self._return_loader(
            **loader_args, X=X_val, y=y_val)
        test_loader = self._return_loader(
            **loader_args, X=X_test, y=y_test)
        test_loader_ID = self._return_loader(
            **loader_args, X=X_test_ID, y=y_test_ID)

        return {
            'train': train_loader,
            'val': validation_loader,
            'test': test_loader,
            'test_ID': test_loader_ID,
            'info': info
        }

    def load_data_subject_inverse(self,
                                  X: np.ndarray,
                                  y: np.ndarray,
                                  OOD_class: list,
                                  seed: int,
                                  discard: None | str
                                  ) -> dict[DataLoader, DataLoader,
                                            DataLoader, DataLoader,
                                            DataLoader, DataLoader,
                                            DataLoader, dict]:
        """
        Pipeline to process (label conversion, z-normalisation, OOD-ID
        splitting and concatenation, etc.) the loaded dataset provided that
        two classes are marked as OOD. Used in the inversion experiment.

        Args:
            X (np.ndarray): Samples.
            y (np.ndarray): Labels belonging to provided samples.
            OOD_class (list): A list of names of classes marked as OOD.
            seed (int): Controls randomness.
            discard (None | str): Optional class to discard.

        Raises:
            ValueError: Raised if more than two classes are marked as OOD.

        Returns:
            dict[DataLoader, DataLoader, DataLoader, DataLoader, DataLoader,
                DataLoader, DataLoader, dict]: The train (ID), validation 1
                (ID), validation 2 (OOD class 1 + ID), validation 3
                (OOD class 2 + ID), test (both ID and OOD data),
                ID_test loaders and dataset metadata dict.
        """

        if len(OOD_class) > 2:
            raise ValueError("Currently only two OOD classes are supported")
        elif len(OOD_class) == 1:
            print("Only 1 class recognised")
            return self.load_data_subject_normal(
                X, y, OOD_class=OOD_class[0], seed=seed, discard=discard
                )

        X, y = self._discard(X, y, discard)

        first_OOD_y = int(len(np.unique(np.array(y))) - 1)
        OOD_data = []

        for OOD_class_i in OOD_class:
            X, y, X_OOD, y_OOD = self._return_OOD_split(
                OOD_class_i, first_OOD_y, X, y
                )
            OOD_data.append((X_OOD, y_OOD))
            print(f"Added OOD class {OOD_class_i} with label {first_OOD_y}")
            first_OOD_y -= 1

        y = self._return_int_labels(y)

        [X_test, y_test, X_val, y_val, X_train, y_train] = (
            self._return_split(X, y, seed=seed, split=2)
            )

        mean, std = self._return_mean_std(X_train)
        loader_args = dict(mean=mean, std=std, data_info=False, shuffle=False)

        train_loader, info = self._return_loader(
            mean, std, X_train, y_train, data_info=True, shuffle=True)
        validation_loader_1 = self._return_loader(
            **loader_args, X=X_val, y=y_val,)

        X_OOD, y_OOD = OOD_data[0]
        X_val_2_1, y_val_2_1 = self._return_split_concat(
            X_val, y_val, X_OOD, y_OOD
            )

        validation_loader_2 = self._return_loader(
            **loader_args, X=X_val_2_1, y=y_val_2_1
            )

        X_OOD, y_OOD = OOD_data[1]
        X_val_2_2, y_val_2_2 = self._return_split_concat(
            X_val, y_val, X_OOD, y_OOD
            )

        validation_loader_3 = self._return_loader(
            **loader_args, X=X_val_2_2, y=y_val_2_2)

        X_OOD, y_OOD = OOD_data[0]
        X_test_2, y_test_2 = self._return_split_concat(
            X_test, y_test, X_OOD, y_OOD
            )
        test_loader = self._return_loader(
            **loader_args, X=X_test_2, y=y_test_2
            )

        X_OOD, y_OOD = OOD_data[1]
        X_test_3, y_test_3 = self._return_split_concat(
            X_test, y_test, X_OOD, y_OOD
            )
        test_loader2 = self._return_loader(
            **loader_args, X=X_test_3, y=y_test_3
            )

        test_loader_ID = self._return_loader(
            **loader_args, X=X_test, y=y_test
            )

        print("Succefully loaded data. \
              Note that validation set 1 is used for model selection. \
              Validation set 2 and 3 are used for determining inversion. \
              Test_k1 and test_k2 contain different OOD classes and are used" \
              "to evaluate OOD detectability.")

        return {
            'train': train_loader,
            'val': validation_loader_1,
            'val_k1': validation_loader_2,
            'val_k2': validation_loader_3,
            'test_k1': test_loader,
            'test_k2': test_loader2,
            'test_ID': test_loader_ID,
            'info': info
        }
