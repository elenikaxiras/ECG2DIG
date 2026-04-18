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

# ===========
# Helpers
# ===========

def metrics_at_threshold(y_true_binary, y_score, threshold, label="positive"):
    """
    Report sensitivity, specificity, PPV, NPV, and counts at a given threshold.
    Prints clear reasons when any metric is undefined (division by zero).
    """
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred, labels=[0, 1]).ravel()
    n = len(y_true_binary)

    def safe_div(num, den, name, reason_if_undef):
        if den == 0:
            return float('nan'), reason_if_undef
        return num / den, None

    reasons = []

    sens, r = safe_div(tp, tp + fn, "Sensitivity",
                       f"Sensitivity is undefined — no positives in y_true "
                       f"(TP+FN = 0); nothing to recall")
    if r: reasons.append(r)

    spec, r = safe_div(tn, tn + fp, "Specificity",
                       f"Specificity is undefined — no negatives in y_true "
                       f"(TN+FP = 0); nothing to rule out")
    if r: reasons.append(r)

    ppv, r = safe_div(tp, tp + fp, "PPV",
                      f"PPV is undefined — model predicts no positives at "
                      f"threshold {threshold:.3f} (TP+FP = 0); no cases to "
                      f"check precision on")
    if r: reasons.append(r)

    npv, r = safe_div(tn, tn + fn, "NPV",
                      f"NPV is undefined — model predicts no negatives at "
                      f"threshold {threshold:.3f} (TN+FN = 0); everything "
                      f"flagged positive")
    if r: reasons.append(r)

    prev = (tp + fn) / n if n else float('nan')

    def fmt(v):
        return f"{v:.3f}" if not np.isnan(v) else "  nan"

    print(f"\n{label} @ threshold={threshold:.3f}:")
    print(f"  Prevalence:  {fmt(prev)}   (n={tp+fn}/{n})")
    print(f"  Sensitivity: {fmt(sens)}   (TP={tp}, FN={fn})")
    print(f"  Specificity: {fmt(spec)}   (TN={tn}, FP={fp})")
    print(f"  PPV:         {fmt(ppv)}")
    print(f"  NPV:         {fmt(npv)}")

    if reasons:
        print("  Notes:")
        for r in reasons:
            print(f"    - {r}")

    return {"threshold": threshold, "sensitivity": sens, "specificity": spec,
            "ppv": ppv, "npv": npv, "prevalence": prev,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}

def threshold_for_target_specificity(y_true_binary, y_score, target_spec=0.95):
    """
    Find the threshold achieving at least `target_spec` specificity,
    with the highest possible sensitivity among such thresholds.
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y_true_binary, y_score)
    spec = 1 - fpr
    # pick the smallest threshold that still meets target_spec
    valid = spec >= target_spec
    if not valid.any():
        return None
    # among valid, take the one with max TPR
    idx = np.where(valid)[0]
    best = idx[np.argmax(tpr[idx])]
    return float(thr[best])

def subgroup_analysis(full_df, target_class=2, threshold=None, target_spec=0.95):
    """
    Report metrics for the target class (e.g., supra = class 2) broken down
    by sex and age group.

    full_df must have columns: 'female', 'age_at_ecg_days', 'dig_class_true',
    and 'cls_logits'.
    """
    probs = softmax(np.vstack(full_df["cls_logits"].values), axis=1)
    y_score = probs[:, target_class]
    y_true = (full_df["dig_class_true"].to_numpy() == target_class).astype(int)

    # If no threshold given, derive one from the overall cohort
    if threshold is None:
        threshold = threshold_for_target_specificity(y_true, y_score, target_spec)
        if threshold is None:
            print(f"No threshold achieves specificity >= {target_spec}; "
                  f"falling back to 0.5")
            threshold = 0.5
        else:
            print(f"Derived threshold for specificity >= {target_spec}: "
                  f"{threshold:.3f}")

    print("\n" + "=" * 60)
    print(f"OVERALL (class {target_class} vs. rest)")
    print("=" * 60)
    metrics_at_threshold(y_true, y_score, threshold, label="Overall")

    # --- sex subgroups ---
    print("\n" + "=" * 60)
    print("BY SEX")
    print("=" * 60)
    for sex_val, label in [(1, "Female"), (0, "Male")]:
        mask = (full_df["female"] == sex_val).to_numpy()
        if mask.sum() == 0 or y_true[mask].sum() == 0:
            print(f"\n{label}: insufficient data (n={int(mask.sum())}, "
                  f"positives={int(y_true[mask].sum())})")
            continue
        _report_group(y_true[mask], y_score[mask], threshold, label, target_class)

    # --- age subgroups (convert days → years inline for thresholds/labels) ---
    print("\n" + "=" * 60)
    print("BY AGE")
    print("=" * 60)
    age_years = full_df["age_at_ecg_days"].to_numpy() / 365.25
    for label, mask in [
        ("< 65 years", age_years < 65),
        (">= 65 years", age_years >= 65),
        (">= 75 years", age_years >= 75),
    ]:
        if mask.sum() == 0 or y_true[mask].sum() == 0:
            print(f"\n{label}: insufficient data (n={int(mask.sum())}, "
                  f"positives={int(y_true[mask].sum())})")
            continue
        _report_group(y_true[mask], y_score[mask], threshold, label, target_class)

    return threshold

def _report_group(y_true_g, y_score_g, threshold, label, target_class):
    """AUC plus operating-point metrics for a subgroup."""
    print(f"\n--- {label} (n={len(y_true_g)}) ---")
    # AUC only if both classes present
    if len(np.unique(y_true_g)) > 1:
        auc = roc_auc_score(y_true_g, y_score_g)
        ap = average_precision_score(y_true_g, y_score_g)
        print(f"  AUC: {auc:.3f}   AP: {ap:.3f}")
    else:
        print(f"  AUC/AP: only one class present in this subgroup")
    metrics_at_threshold(y_true_g, y_score_g, threshold, label=label)
    
def plot_roc_curves(full_df, class_names=None):
    if class_names is None:
        class_names = {0: "No digoxin", 1: "Therapeutic", 2: "Supra"}

    probs = softmax(np.vstack(full_df["cls_logits"].values), axis=1)
    y_true = full_df["dig_class_true"].to_numpy()
    C = probs.shape[1]

    fig, ax = plt.subplots(figsize=(6, 6))
    for c in range(C):
        y_bin = (y_true == c).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        fpr, tpr, _ = roc_curve(y_bin, probs[:, c])
        ax.plot(fpr, tpr, label=f"{class_names[c]} (AUC={sk_auc(fpr,tpr):.3f})")
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves (one-vs-rest)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

def compute_f1(y_true, y_pred, C):
    """
    Returns (per_class_f1_dict, macro_f1, reasons_list).
    Any class that can't be meaningfully evaluated gets a reason string
    instead of silently becoming NaN.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    true_counts = np.bincount(y_true, minlength=C)
    pred_counts = np.bincount(y_pred, minlength=C)

    per_class_f1 = {}
    reasons = []

    # Per-class F1 with zero_division=0 so we get numbers instead of NaN,
    # but we still flag the degenerate cases for the user.
    f1s = f1_score(y_true, y_pred, average=None,
                   labels=np.arange(C), zero_division=0)

    for c in range(C):
        f1_c = float(f1s[c])
        per_class_f1[c] = f1_c

        if true_counts[c] == 0 and pred_counts[c] == 0:
            reasons.append(f"class {c}: absent from both y_true and y_pred "
                           f"→ F1 set to 0 (undefined)")
        elif true_counts[c] == 0:
            reasons.append(f"class {c}: no true samples "
                           f"(predicted {pred_counts[c]} times) → F1 = 0")
        elif pred_counts[c] == 0:
            reasons.append(f"class {c}: never predicted "
                           f"({true_counts[c]} true samples) → F1 = 0")

    macro_f1 = float(np.mean(f1s))

    # Macro F1 is a valid number, but warn if it's dominated by "free zeros"
    degenerate = sum(1 for c in range(C)
                     if true_counts[c] == 0 and pred_counts[c] == 0)
    if degenerate > 0:
        reasons.append(f"macro F1 averages over {C} classes, but "
                       f"{degenerate} had no samples at all "
                       f"(inflating toward 0)")

    return per_class_f1, macro_f1, reasons

def _parse_scalar(x: str):
    x = x.strip()
    if x in {"True", "False"}:
        return x == "True"
    try:
        return ast.literal_eval(x)
    except Exception:
        return x

def parse_args_from_training_log(log_path):
    lines = Path(log_path).read_text().splitlines()

    args = {}
    in_args = False
    for line in lines:
        if line.strip() == "ARGS:":
            in_args = True
            continue

        if in_args:
            if not line.startswith("  "):
                break
            key, value = line.strip().split(":", 1)
            args[key] = _parse_scalar(value)

    return args

def _extract_state_dict(ckpt):
    """
    Handles:
      1) raw state_dict
      2) {'model_state_dict': ...}
      3) {'state_dict': ...}
      4) DDP/DataParallel keys with 'module.' prefix
    """
    if not isinstance(ckpt, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(ckpt)}")

    if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state_dict = ckpt["state_dict"]
    elif all(torch.is_tensor(v) for v in ckpt.values()):
        state_dict = ckpt
    else:
        raise ValueError(
            "Could not find a valid state_dict in checkpoint. "
            f"Top-level keys: {list(ckpt.keys())[:20]}"
        )

    # Strip DDP/DataParallel prefix if present
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {
            k.replace("module.", "", 1): v
            for k, v in state_dict.items()
        }

    return state_dict

def load_model_from_weights(model_path, model_kwargs, device=None, strict=True):
    """
    model_path: path to model_best_state_dict.pt
    model_kwargs: dict with the exact args needed to build the model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # 1) build model from ecg2dig code
    model = ECG2DIG(**model_kwargs).to(device)

    # 2) load checkpoint
    ckpt = torch.load(model_path, map_location=device)

    # 3) extract weights
    state_dict = _extract_state_dict(ckpt)

    # 4) load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)

    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)

    # 5) inference mode
    model.eval()
    return model

def build_model_kwargs_from_log(log_path):
    args = parse_args_from_training_log(log_path)

    model_kwargs = {
        "name": "ECG2DIG",
        "num_classes": args["num_classes"],
        "dig_threshold": args["dig_threshold"],
    }
    return model_kwargs
