"""
bend.models
===========

This module contains the implementations of the supervised models used in the paper.

- :class:`~bend.models.downstream.ConvNetForSupervised`: a ResNet that we train as baseline model on one-hot encodings, if no dedicated baseline architecture is available for a task.
- :class:`~bend.models.downstream.CNN`: a two-layer CNN used for all downstream tasks.
"""

from .awd_lstm import AWDLSTMConfig, AWDLSTMForLM, AWDLSTMModelForInference
from .dilated_cnn import ConvNetConfig, ConvNetForMaskedLM, ConvNetModel
from .downstream import CNN, ConvNetForSupervised
