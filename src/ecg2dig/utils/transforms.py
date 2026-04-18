from __future__ import annotations
import os
import json
import sys
import re
import copy
import ast
import random
from pathlib import Path
import math
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from scipy.special import softmax
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sk_auc
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from ml4h.defines import PARTNERS_DATETIME_FORMAT, ECG_REST_AMP_LEADS
from ml4h.TensorMap import TensorMap, Interpretation
import ecg2dig 
# from ecg2dig.utils.datasets import ECGDrugDataset, ECGDataDescription
# from ecg2dig.utils.helpers import load_model_from_weights, build_model_kwargs_from_log
# from ecg2dig.utils.helpers import parse_args_from_training_log
from ecg2dig.ECG2DIG import ECG2DIG
# ----------
# ECG Transformations
# ----------

class ScaleECGTransform:
    """
    Transform ECG units to millivolts
    If we want a copy: return torch.tensor(ecg, dtype=torch.float)
    """
    def __call__(self, ecg: torch.Tensor,
                        ) -> torch.Tensor:
        return (ecg/1000)
    
class ToTensorTransform:   
    """
    Convert input to a torch.FloatTensor with minimal extra copies:
      - If it's already a Tensor and float32: reused in place.
      - If it's a Tensor but not float32: one cast to float32.
      - If it's a NumPy array: one copy via astype(np.float32) + from_numpy.
    """
    def __call__(self, ecg) -> torch.Tensor:
        # Case 1: already a torch.Tensor
        if isinstance(ecg, torch.Tensor):
            # If it's already float32, return directly (no copy).
            if ecg.dtype == torch.float32:
                return ecg
            # Otherwise cast to float32 (this will allocate a new buffer).
            return ecg.to(torch.float32)

        # Case 2: assume it's a NumPy array
        # astype(np.float32, copy=False) will only copy if dtype != float32.
        arr32 = ecg.astype(np.float32, copy=False)
        return torch.from_numpy(arr32)

class StandardizeByChannelECGTransform:
    """ Standardizes an ECG sample on a per-channel basis by computing
    the mean and standard deviation for each channel (over the T time points)
    Args:
        - ECG tensor of shape [T, C] (e.g. [5000, 12]).
    Returns:
        - ECG PyTorch Tensor
    """
    def __init__(self, eps=1e-6):
        self.eps = eps

    def __call__(self, ecg: torch.Tensor,
                ) -> torch.Tensor:
        channel_mean = torch.mean(ecg, dim=0, keepdim=True)
        channel_std = torch.std(ecg, dim=0, keepdim=True)
        return (ecg - channel_mean) / (channel_std + self.eps)
class ClipECGByPercentileTransform:
    """
    Clip each channel of an ECG by a per-channel threshold estimated via torch.quantile,
    avoiding any numpy/SciPy calls so that DataLoader workers clean up nicely.

    Args:
        clip_factor (float): multiply the per-channel percentile by this (default 1.5)
        pct (float): which percentile of abs(signal) to use, between 0 and 100 (default 99.5)
    """
    def __init__(self, clip_factor: float = 1.5, pct: float = 99.5):
        self.clip_factor = clip_factor
        self.q = pct / 100.0

    def __call__(self, ecg: torch.Tensor) -> torch.Tensor:
        # ecg can be [T, C] or [C, T]
        x = ecg
        flipped = False
        if x.ndim == 2 and x.shape[0] <= 12:   # assume [C, T]
            x = x.transpose(0, 1)
            flipped = True

        # x is now [T, C]; compute per-channel percentile of abs(x)
        # torch.quantile will preserve dtype (e.g. float32 or float16)
        thresh = torch.quantile(x.abs(), self.q, dim=0, keepdim=True)  # [1, C]
        thr = thresh * self.clip_factor

        # clamp and restore original orientation
        x = torch.clamp(x, -thr, thr)
        if flipped:
            x = x.transpose(0, 1)
        return x   
    
# --------
# Feature Transformations (non-ECG)
# --------

class StandardizeTransform:
    def __init__(self, feature_means, feature_stds):
        """
        Standardize many features at the same time
        Used for interval standardize
        feature_means and feature_stds should be dictionaries mapping
        feature names to their respective mean and std.
        
        Guarding against division by sero if feature is constant
        """
        self.feature_means = feature_means
        self.feature_stds = feature_stds
    
    #  Guarding against division by sero if feature is constant
    def __call__(self, sample):
        for feature, mean in self.feature_means.items():
            if feature in sample:
                std = self.feature_stds[feature]
                if std == 0:
                    continue  # leave sample[feature] unchanged
                sample[feature] = (sample[feature] - mean) / std
        return sample
    
    def inverse(self, sample): # .inverse(sample.copy())
        """
        Inverse transform a sample dictionary that was standardized.
        Returns a new sample dict with the specified features un-standardized.
        NOTE: sample should be passed as sample.copy() to avoid altering
        the original
        """
        
        for feature, mean in self.feature_means.items():
            if feature in sample:
                std = self.feature_stds[feature]
                sample[feature] = sample[feature] * std + mean
        return sample
    

