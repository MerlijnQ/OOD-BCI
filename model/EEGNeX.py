import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, remove_spectral_norm
from torch.nn.utils.parametrize import register_parametrization
import math


class ConvSpectral(nn.Module):
    """
    Custom 2D convolutional layer with optional spectral normalization.

    This layer wraps a standard `nn.Conv2d` and allows spectral
    normalization to be applied or removed dynamically.

    Attributes:
        conv (nn.Conv2d): The underlying convolutional layer.
        sn_applied (bool): Flag indicating whether spectral normalization
            has been applied to the layer.
    """
    def __init__(self,
                 f_in: int,
                 f_out: int,
                 k: int,
                 padding: int = 0,
                 bias: bool = False,
                 dilation: int = 1,
                 groups: int = 1,
                 **kwargs
                 ) -> None:
        """
        Initialize the ConvSpectral layer.

        Args:
            f_in (int): Number of input channels.
            f_out (int): Number of output channels.
            k (int): Kernel size.
            padding (int, optional): Padding added to all sides of the input.
                Defaults to 0.
            bias (bool, optional): If True, adds a learnable bias to the
                output. Defaults to False.
            dilation (int, optional): Spacing between kernel elements.
                Defaults to 1.
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Defaults to 1.
            **kwargs: Additional keyword arguments passed to `nn.Conv2d`
                (e.g., `stride`, `padding_mode`, etc.).
        """
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=f_in,
            out_channels=f_out,
            kernel_size=k,
            padding=padding,
            bias=bias,
            dilation=dilation,
            groups=groups,
            **kwargs
            )

        self.sn_applied = False

    def apply_spectral(self):
        """ Applying spectral normalisation to the convolutional
        layer if not yet applied.
        """
        if not self.sn_applied:
            self.conv = spectral_norm(self.conv)
            self.sn_applied = True

    def remove_spectral(self):
        """Removing spectral normalisation from the convolutional
        layer is it was applied.
        """
        if self.sn_applied:
            remove_spectral_norm(self.conv)
            self.sn_applied = False

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """A forward pass through the convolutional layer

        Args:
            X (torch.Tensor): An input vector.

        Returns:
            torch.Tensor: The result of the forward pass.
        """
        return self.conv(X)


class MaxNormEnforced(nn.Module):
    """
    Module that enforces a maximum L2 norm on layer weights.

    This module is intended to be applied to weight tensors in model layers 
    to constrain their L2 norm to a specified maximum value.
    The implementation is adapted from Braindecode:
    https://github.com/braindecode/braindecode/blob/master/braindecode/modules/parametrization.py#L27

    Attributes:
        max_norm (float): Maximum L2 norm allowed for the weights.
    """
    def __init__(self,
                 max_norm: float = 1.0
                 ) -> None:
        """
        Initialize the MaxNormEnforced module.

        Args:
            max_norm (float, optional): The maximum allowed L2 norm for the
                weights. Defaults to 1.0.
        """
        super().__init__()
        self.max_norm = max_norm

    def forward(self,
                X: torch.Tensor
                ) -> torch.Tensor:
        """Performs the operation of renorming the weight tensor over dim=0.

        Args:
            X (torch.Tensor): A weight tensor on which the norm should
            be enforced.

        Returns:
            torch.Tensor: The renormed weight tensor.
        """
        return X.renorm(p=2, dim=0, maxnorm=self.max_norm)


class Conv2dWithConstraint(nn.Conv2d):
    """
    2D convolutional layer with L2 norm constraint on weights.

    This layer extends `nn.Conv2d` and enforces a maximum L2 norm on the weight
    tensor using a parametrization. Weight initialization is performed using
    Xavier uniform initialization.
    Adapted from Braindecode:
    https://github.com/braindecode/braindecode/blob/master/braindecode/modules/convolution.py#L75

    Attributes:
        max_norm (float): Maximum allowed L2 norm for the convolution weights.
    """
    def __init__(self,
                 *args: dict,
                 max_norm: float = 1.0,
                 **kwargs: dict) -> None:
        """
        Initialize the Conv2dWithConstraint layer.

        Args:
            *args: Positional arguments for 'nn.Conv2d'.
            max_norm (float, optional): Maximum allowed L2 norm for the
                weights. Defaults to 1.0.
            **kwargs: Keyword arguments for 'nn.Conv2d'.
        """
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm
        nn.init.xavier_uniform_(self.weight, gain=1.0)
        register_parametrization(self, "weight", MaxNormEnforced(max_norm))


class LinearWithConstraint(nn.Linear):
    """
    Linear (dense) layer with optional L2 norm constraint on weights.

    This layer extends 'nn.Linear' and enforces a maximum L2 norm on the weight
    tensor using a parametrization.
    Adapted from Braindecode:
    https://github.com/braindecode/braindecode/blob/master/braindecode/modules/linear.py#L43

    Attributes:
        max_norm (float): Maximum allowed L2 norm for the layer's weights.
    """
    def __init__(self,
                 *args: dict,
                 max_norm: float = 0.25,
                 **kwargs: dict
                 ) -> None:
        """
        Initialize the LinearWithConstraint layer.

        Args:
            *args: Positional arguments for `nn.Linear`.
            max_norm (float, optional): Maximum allowed L2 norm for the
                weights. Defaults to 0.25.
            **kwargs: Keyword arguments for `nn.Linear`.
        """
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm
        register_parametrization(self, "weight", MaxNormEnforced(max_norm))


class EEGNeX_8_32(nn.Module):
    """
    PyTorch implementation of the EEGNeX model for EEG classification.

    - Original paper: https://doi.org/10.1016/j.bspc.2023.105475
    - Keras implementation: https://github.com/chenxiachan/EEGNeX

    The model consists of several convolutional and depthwise convolutional
    layers, followed by spectral-normalized penultimate features, which can
    be useful for OOD detection methods that operate in feature space.

    Attributes:
        linear_size (int): Size of the flattened feature vector before the
            final linear layer.
    """
    def __init__(self,
                 n_classes: int,
                 n_timesteps: int,
                 n_channels: int = 1,
                 drop_prob: float = 0.5,
                 affine: bool = True
                 ) -> None:
        """
        Initialize the EEGNeX model.

        Args:
            n_classes (int): Number of output classes.
            n_timesteps (int): Number of timesteps in each input sample.
            n_channels (int, optional): Number of EEG channels. Defaults to 1.
            drop_prob (float, optional): Dropout probability. Defaults to 0.5.
            affine (bool, optional): Whether batch normalization layers are
                learnable. Defaults to True.
        """
        super().__init__()

        k, pad = 4, 1

        T3 = math.floor((n_timesteps + 2 * pad - k) / k) + 1
        T5 = math.floor((T3 + 2 * pad - k) / k) + 1

        self.linear_size = 8 * T5

        # layer 1
        self.conv1 = nn.Conv2d(1, 8, (1, 64), padding='same', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(8, affine=affine)

        # Layer 2
        self.conv2 = nn.Conv2d(8, 32, (1, 64), padding='same', bias=False)
        self.batchnorm2 = nn.BatchNorm2d(32, affine=affine)

        # layer 3
        self.depthwise3 = Conv2dWithConstraint(in_channels=32,
                                               out_channels=64,
                                               kernel_size=(n_channels, 1),
                                               groups=32,
                                               bias=False)
        self.batchnorm3 = nn.BatchNorm2d(64, affine=affine)
        self.pool3 = nn.AvgPool2d((1, 4),  padding=(0, 1))
        self.dropout3 = nn.Dropout(drop_prob)

        # Layer 4
        self.conv4 = nn.Conv2d(
            64, 32, (1, 16), padding='same', dilation=(1, 2), bias=False
            )
        self.batchnorm4 = nn.BatchNorm2d(32, affine=affine)

        # Layer 5
        self.conv5 = ConvSpectral(
                32, 8, (1, 16), padding='same', dilation=(1, 4), bias=False
            )
        self.batchnorm5 = nn.BatchNorm2d(8, affine=affine)
        self.pool5 = nn.AvgPool2d(
                (1, 4),  padding=(0, 1)
            )
        self.dropout5 = nn.Dropout(drop_prob)

        # Layer 6
        self.flat6 = nn.Flatten()

        self.linear6 = LinearWithConstraint(self.linear_size, n_classes)

    def feature_extractor(self,
                          x: torch.Tensor
                          ) -> torch.Tensor:
        """
        Performs a forward pass through the EEGNeX model until the
        penultimate layer.

        Args:
            x (torch.Tensor): A raw input.

        Returns:
            torch.Tensor: A feature vector.
        """

        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.elu(self.depthwise3(x))
        x = self.batchnorm3(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = F.elu(self.conv5(x))
        x = self.batchnorm5(x)
        x = self.pool5(x)
        x = self.dropout5(x)
        penultimate_vector = self.flat6(x)

        return penultimate_vector

    def classify(self,
                 x: torch.Tensor
                 ) -> torch.Tensor:
        """
        Classifies a sample from a feature vector.

        Args:
            x (torch.Tensor): A feature vector.

        Returns:
            torch.Tensor: A vector in logit space which
                can be used for softmax classification of a sample.
        """
        return self.linear6(x)

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        """
        A full forward pass through the EEGNeX model.

        Args:
            x (torch.Tensor): A raw input tensor.

        Returns:
            torch.Tensor: A vector in logit space which
                can be used for softmax classification of a sample.
        """
        x = self.feature_extractor(x)
        fc = self.classify(x)
        return fc
