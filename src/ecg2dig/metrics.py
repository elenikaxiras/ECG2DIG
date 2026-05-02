import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from scipy.special import softmax
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    f1_score, balanced_accuracy_score, matthews_corrcoef,
    confusion_matrix, cohen_kappa_score
)
from sklearn.calibration import calibration_curve
import torch
import torch.nn.functional as F

# -----------------------------
# Basics
# -----------------------------
def _softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    ex = np.exp(logits)
    return ex / np.sum(ex, axis=1, keepdims=True)

def prepare_probs_targets_df(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        probs: (N,3) softmax probabilities
        y    : (N,)  int labels in {0,1,2}
        logits: (N,3) raw logits
    """
    logits = np.stack(df["cls_logits"].apply(lambda v: np.array(v, dtype=float)))
    y = df["cls_true"].astype(int).values
    # drop rows with NaN labels (if any)
    mask = ~np.isnan(y) if np.issubdtype(y.dtype, np.floating) else np.ones_like(y, dtype=bool)
    logits = logits[mask]
    y = y[mask].astype(int)
    probs = _softmax_np(logits)
    return probs, y, logits

# -----------------------------
# Metrics (multiclass)
# -----------------------------
def evaluate_multiclass_metrics_df(df: pd.DataFrame, verbose: bool = False):
    probs, y, _ = prepare_probs_targets_df(df)
    preds = probs.argmax(axis=1)

    if verbose:
        print(f"[eval] N={len(y)}")

    # Macro/weighted AUC (one-vs-rest)
    try:
        auc_macro = roc_auc_score(y, probs, multi_class="ovr", average="macro")
        auc_weighted = roc_auc_score(y, probs, multi_class="ovr", average="weighted")
    except ValueError:
        auc_macro = np.nan
        auc_weighted = np.nan

    # Per-class ROC/PR
    per_class = {}
    C = probs.shape[1]
    for c in range(C):
        y_bin = (y == c).astype(int)
        if len(np.unique(y_bin)) > 1:
            auc = roc_auc_score(y_bin, probs[:, c])
            ap  = average_precision_score(y_bin, probs[:, c])
        else:
            auc, ap = np.nan, np.nan
        per_class[f"class{c}"] = {"AUC": auc, "AP": ap}
        if verbose:
            pos = int(y_bin.sum())
            auc_str = "nan" if np.isnan(auc) else f"{auc:.3f}"
            ap_str  = "nan" if np.isnan(ap)  else f"{ap:.3f}"
            print(f"[eval] class {c}: positives={pos}, AUC={auc_str}, AP={ap_str}")


    metrics = dict(
        auc_macro=auc_macro,
        auc_weighted=auc_weighted,
        balanced_accuracy=balanced_accuracy_score(y, preds),
        macro_f1=f1_score(y, preds, average="macro"),
        micro_f1=f1_score(y, preds, average="micro"),
        mcc=matthews_corrcoef(y, preds),
        qwk=cohen_kappa_score(y, preds, weights="quadratic"),
        confusion_matrix=confusion_matrix(y, preds).tolist(),
        per_class=per_class
    )
    return metrics, probs, y

# -----------------------------
# Decision Curve Analysis (class 2 vs {0,1})
# -----------------------------
def plot_decision_curve_analysis_df(df: pd.DataFrame, positive_class: int = 2):
    probs, y, _ = prepare_probs_targets_df(df)
    y_bin = (y == positive_class).astype(int)
    p = probs[:, positive_class]
    N = len(y_bin)
    prevalence = y_bin.mean()

    thresholds = np.linspace(0.01, 0.99, 99)
    nb_model, nb_all, nb_none = [], [], []

    for pt in thresholds:
        pred = (p >= pt).astype(int)
        TP = np.sum((pred == 1) & (y_bin == 1))
        FP = np.sum((pred == 1) & (y_bin == 0))
        nb_model.append((TP / N) - (FP / N) * (pt / (1.0 - pt)))
        nb_all.append(prevalence - (1.0 - prevalence) * (pt / (1.0 - pt)))
        nb_none.append(0.0)

    plt.figure()
    plt.plot(thresholds, nb_model, label=f"Model (class {positive_class} vs others)")
    plt.plot(thresholds, nb_all, label="Treat-all")
    plt.plot(thresholds, nb_none, label="Treat-none")
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title(f"Decision Curve Analysis – class {positive_class} vs others")
    plt.legend()
    plt.show()

# -----------------------------
# O-v-R ROC & PR (all classes on one figure each)
# -----------------------------
def plot_multiclass_roc_df(df: pd.DataFrame, 
                           class_names=("Class 0","Class 1","Class 2"),
                           save_path = None,
                           legend_title=None,
                           dpi=600,
                           res=None):
    probs, y, _ = prepare_probs_targets_df(df)
    
    # Get tab20 colors
    cmap = plt.cm.tab20
    colors = cmap([0, 4, 6])
    
    plt.figure(dpi=600)
    for c in range(probs.shape[1]):
        y_bin = (y == c).astype(int)
        if len(np.unique(y_bin)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_bin, probs[:, c])
        auc = roc_auc_score(y_bin, probs[:, c])
        if res:
            ci = res["per_class"][f"class{c}"]["AUC_CI"]
            label = f"{class_names[c]}   {auc:.2f} ({ci[0]:.2f}-{ci[1]:.2f})"
        else:
            label = f"{class_names[c]}   {auc:.2f}"
        plt.plot(fpr, tpr, label=label, color=colors[c], linewidth=2)
    
    plt.plot([0,1],[0,1],"--", alpha=0.6, color='gray')
    plt.xlabel("1 - Specificity", fontweight='bold', fontsize=10)
    plt.ylabel("Sensitivity", fontweight='bold', fontsize=10)
    #plt.title("Receiver-Operating Characteristic Curves for One-vs-Rest")
    legend = plt.legend(title=legend_title, 
                        title_fontproperties={'weight': 'bold', 'size': 8},
                        fontsize=8,
                        loc='lower right')
    
    legend._legend_box.align = "right"
    legend.get_title().set_ha('right')
    
    if save_path:
        plt.savefig(" ".join([save_path, ".tiff"]), dpi=dpi, format="tiff", bbox_inches="tight")
        plt.savefig(" ".join([save_path, ".pdf"]), dpi=dpi, format="pdf", bbox_inches="tight")
        
    plt.show()


def plot_multiclass_pr_df(df: pd.DataFrame, 
                          class_names=("Class 0","Class 1","Class 2"),
                          save_path = None,
                          legend_title=None,
                          dpi=600,
                          res=None):
    probs, y, _ = prepare_probs_targets_df(df)
    
    # Get tab20 colors
    cmap = plt.cm.tab20
    colors = cmap([0, 4, 6])
    
    # Calculate padding for alignment
    max_name_len = max(len(name) for name in class_names)
    
    plt.figure(dpi=300)
    for c in range(probs.shape[1]):
        y_bin = (y == c).astype(int)
        if len(np.unique(y_bin)) < 2:
            continue
        prec, rec, _ = precision_recall_curve(y_bin, probs[:, c])
        ap = average_precision_score(y_bin, probs[:, c])
        if res:
            ci = res["per_class"][f"class{c}"]["PR_AUC_CI"]
            label = f"{class_names[c]} {ap:.2f} ({ci[0]:.2f}-{ci[1]:.2f})"
        else:
            label = f"{class_names[c]} {ap:.2f}"
        
        line = plt.plot(rec, prec, label=label, color=colors[c], linewidth=2)[0]
        plt.hlines(y_bin.mean(), xmin=0, xmax=1, 
                   colors=line.get_color(), linestyles="dotted", alpha=0.6)
        
    plt.xlabel("Recall", fontweight='bold', fontsize=10)
    plt.ylabel("Precision", fontweight='bold', fontsize=10)
    legend = plt.legend(title=legend_title, 
                       title_fontproperties={'weight': 'bold', 'size': 8},
                       loc='center right',
                       bbox_to_anchor=(1, 0.7),  
                       fontsize=8,
                       ) 
    legend._legend_box.align = "right"
    legend.get_title().set_ha('left')
    plt.xlim(0,1); 
    plt.ylim(0,1)
    if save_path:
        plt.savefig(" ".join([save_path, ".tiff"]), dpi=dpi, format="tiff", bbox_inches="tight")
        plt.savefig(" ".join([save_path, ".pdf"]), dpi=dpi, format="pdf", bbox_inches="tight")
    plt.show()
    

# ---------------------------
# O-v-O AUC and PR
# -----------------------------

def plot_ovo_roc_df(df: pd.DataFrame, 
                    pairs=((0,1),(0,2),(1,2)),
                    pair_names=("0 vs 1","0 vs 2","1 vs 2"),
                    save_path=None,
                    legend_title=None,
                    dpi=600,
                    res=None):
    """
    Plot ROC curves for one-vs-one classification pairs.
    """
    probs, y, _ = prepare_probs_targets_df(df)
    plt.figure(dpi=dpi)
    
    # Your color choice from tab20: indices 2, 8, 10
    cmap = plt.cm.tab20
    colors = cmap([2, 8, 10])

    for idx, (a, b) in enumerate(pairs):
        # Filter to only samples from classes a and b
        mask = (y == a) | (y == b)
        if not mask.any():
            continue
        
        y_pair = y[mask]
        probs_pair = probs[mask]
        
        # Binary labels: class b is positive (1), class a is negative (0)
        y_bin = (y_pair == b).astype(int)
        scores = probs_pair[:, b]  # probability of class b

        if len(np.unique(y_bin)) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_bin, scores)
        auc = roc_auc_score(y_bin, scores)

        # Format label with CI if res is provided
        pair_key = f"{a}vs{b}"
        if res and pair_key in res.get("pairs", {}):
            ci = res["pairs"][pair_key]["AUC_CI"]
            label = f"{pair_names[idx]}   {auc:.2f} ({ci[0]:.2f}-{ci[1]:.2f})"
        else:
            label = f"{pair_names[idx]}   {auc:.2f}"

        plt.plot(
            fpr, tpr,
            label=label,
            color=colors[idx % len(colors)],
            linewidth=2
        )
    
    plt.plot([0, 1], [0, 1], "--", alpha=0.6)
    plt.xlabel("1 - Specificity", fontweight='bold', fontsize=10)
    plt.ylabel("Sensitivity", fontweight='bold', fontsize=10)
    
    legend = plt.legend(
        title=legend_title, 
        title_fontproperties={'weight': 'bold', 'size': 8},
        fontsize=8,
        loc='lower right'
    )
    legend._legend_box.align = "right"
    legend.get_title().set_ha('right')
    
    if save_path:
        plt.savefig(f"{save_path}.tiff", dpi=dpi, format="tiff", bbox_inches="tight")
        plt.savefig(f"{save_path}.pdf", dpi=dpi, format="pdf", bbox_inches="tight")
    
    plt.show()

def plot_ovo_pr_df(df: pd.DataFrame, 
                   pairs=((0,1),(0,2),(1,2)),
                   pair_names=("0 vs 1","0 vs 2","1 vs 2"),
                   save_path=None,
                   legend_title=None,
                   dpi=600,
                   res=None):
    """
    Plot Precision-Recall curves for one-vs-one classification pairs.
    """
    probs, y, _ = prepare_probs_targets_df(df)
    plt.figure(dpi=dpi)
    
    # Your color selection from tab20
    cmap = plt.cm.tab20
    colors = cmap([2, 8, 10])
    
    for idx, (a, b) in enumerate(pairs):
        # Filter to only samples from classes a and b
        mask = (y == a) | (y == b)
        if not mask.any():
            continue
        
        y_pair = y[mask]
        probs_pair = probs[mask]
        
        # Binary labels: class b is positive (1), class a is negative (0)
        y_bin = (y_pair == b).astype(int)
        scores = probs_pair[:, b]
        
        if len(np.unique(y_bin)) < 2:
            continue
        
        prec, rec, _ = precision_recall_curve(y_bin, scores)
        ap = average_precision_score(y_bin, scores)
        
        pair_key = f"{a}vs{b}"
        if res and pair_key in res.get("pairs", {}):
            ci = res["pairs"][pair_key]["PR_AUC_CI"]
            label = f"{pair_names[idx]}   {ap:.2f} ({ci[0]:.2f}-{ci[1]:.2f})"
        else:
            label = f"{pair_names[idx]}   {ap:.2f}"
        
        color = colors[idx % len(colors)]
        plt.plot(rec, prec, label=label, color=color, linewidth=2)
        
        # Baseline (prevalence)
        plt.hlines(y_bin.mean(), xmin=0, xmax=1, 
                   colors=color, linestyles="dotted", alpha=0.6)
    
    plt.xlabel("Recall", fontweight='bold', fontsize=10)
    plt.ylabel("Precision", fontweight='bold', fontsize=10)
    
    legend = plt.legend(title=legend_title, 
                        title_fontproperties={'weight': 'bold', 'size': 8},
                        loc='upper right',
                        fontsize=8)
    legend._legend_box.align = "right"
    legend.get_title().set_ha('right')
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    if save_path:
        plt.savefig(f"{save_path}.tiff", dpi=dpi, format="tiff", bbox_inches="tight")
        plt.savefig(f"{save_path}.pdf", dpi=dpi, format="pdf", bbox_inches="tight")
    
    plt.show()



# -----------------------------
def _ece_per_class(probs: np.ndarray, y: np.ndarray, n_bins: int = 10, strategy: str = "quantile"):
    """Per-class ECE with explicit bin sizes."""
    eces = {}
    C = probs.shape[1]
    for c in range(C):
        y_bin = (y == c).astype(int)
        if len(np.unique(y_bin)) < 2:
            eces[f"class{c}"] = np.nan
            continue
        p = probs[:, c]
        if strategy == "uniform":
            bins = np.linspace(0, 1, n_bins + 1)
        else:  # quantile
            bins = np.quantile(p, np.linspace(0, 1, n_bins + 1))
            bins[0], bins[-1] = 0.0, 1.0
        inds = np.digitize(p, bins) - 1
        inds = np.clip(inds, 0, n_bins - 1)
        ece = 0.0
        for k in range(n_bins):
            idx = (inds == k)
            if not np.any(idx):
                continue
            bin_true = y_bin[idx].mean()
            bin_pred = p[idx].mean()
            ece += (idx.mean()) * abs(bin_true - bin_pred)
        eces[f"class{c}"] = ece
    return eces

def evaluate_calibration_df(df: pd.DataFrame, 
                            n_bins: int = 10, 
                            strategy: str = "quantile", 
                            plot: bool = True):
    probs, y, _ = prepare_probs_targets_df(df)
    # Multiclass Brier
    onehot = np.zeros_like(probs); onehot[np.arange(len(y)), y] = 1
    brier = np.mean(np.sum((probs - onehot) ** 2, axis=1))

    # Per-class reliability curves
    curves = {}
    C = probs.shape[1]
    for c in range(C):
        y_bin = (y == c).astype(int)
        if len(np.unique(y_bin)) < 2:
            curves[f"class{c}"] = None
            continue
        prob_true, prob_pred = calibration_curve(y_bin, probs[:, c], n_bins=n_bins, strategy=strategy)
        curves[f"class{c}"] = {"prob_true": prob_true, "prob_pred": prob_pred}

    eces = _ece_per_class(probs, y, n_bins=n_bins, strategy=strategy)

    if plot:
        for c in range(C):
            data = curves.get(f"class{c}")
            if data is None:
                continue
            plt.figure(dpi=300)
            plt.plot(data["prob_pred"], data["prob_true"], marker="o", label=f"class {c}")
            plt.plot([0,1],[0,1], "--", alpha=0.6)
            plt.xlabel("Predicted probability")
            plt.ylabel("Empirical frequency")
            plt.title(f"Reliability – class {c} (ECE={eces[f'class{c}']:.3f})")
            plt.legend(); plt.xlim(0,1); plt.ylim(0,1)
            plt.show()

    return {"brier": brier, "ece_per_class": eces, "curves": curves, "probs": probs, "y": y}


# -----------------------------
# Temperature scaling (optional)
# -----------------------------
# def _nll_from_logits_np(logits: np.ndarray, y: np.ndarray) -> float:
#     l = torch.from_numpy(logits).float()
#     t = torch.from_numpy(y).long()
#     return float(F.cross_entropy(l, t, reduction="mean").item())

# def fit_temperature_on_df(df: pd.DataFrame, max_iter: int = 100, lr: float = 1e-2):
#     """
#     Fits a single temperature T to minimize NLL on the provided df (logits+labels).
#     Returns:
#         calibrate(logits_np) -> calibrated_probs_np
#         nll_before, nll_after
#     """
#     _, y, logits = prepare_probs_targets_df(df)
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     l = torch.from_numpy(logits).to(device).float()
#     t = torch.from_numpy(y).to(device).long()

#     logT = torch.nn.Parameter(torch.zeros(()).to(device))  # T=1
#     opt = torch.optim.LBFGS([logT], lr=lr, max_iter=max_iter)

#     def closure():
#         opt.zero_grad()
#         loss = F.cross_entropy(l / torch.exp(logT), t)
#         loss.backward()
#         return loss

#     opt.step(closure)

#     @torch.no_grad()
#     def calibrate(logits_np: np.ndarray) -> np.ndarray:
#         L = torch.from_numpy(logits_np).to(device).float()
#         p = F.softmax(L / torch.exp(logT), dim=1).cpu().numpy()
#         return p

#     nll_before = _nll_from_logits_np(logits, y)
#     nll_after  = _nll_from_logits_np((logits / np.exp(logT.detach().cpu().numpy())), y)
#     return calibrate, float(nll_before), float(nll_after)

# def apply_and_report_temperature_scaling(df_val, df_test, n_bins=10, strategy="quantile"):
#     """
#     Fits T on df_val, applies to df_test, and PRINTS outcomes:
#       - Val NLL before/after
#       - Test macro AUROC (should be invariant)
#       - Test Brier before/after
#       - Per-class ECE before/after
#       - % of samples whose argmax changed (should be ~0)
#     Returns a dict with the same numbers.
#     """
#     # 1) Fit temperature on validation
#     calibrate, nll_before, nll_after = fit_temperature_on_df(df_val)

#     # 2) Test set: pre-/post- calibration
#     probs_pre, y_test, logits_test = prepare_probs_targets_df(df_test)
#     probs_post = calibrate(logits_test)

#     # 3) Metrics
#     # AUROC (discrimination) – should not change
#     auc_macro_pre  = roc_auc_score(y_test, probs_pre,  multi_class="ovr", average="macro")
#     auc_macro_post = roc_auc_score(y_test, probs_post, multi_class="ovr", average="macro")

#     # Brier (overall calibration)
#     onehot = np.zeros_like(probs_pre); onehot[np.arange(len(y_test)), y_test] = 1
#     brier_pre  = float(np.mean(np.sum((probs_pre  - onehot)**2, axis=1)))
#     brier_post = float(np.mean(np.sum((probs_post - onehot)**2, axis=1)))

#     # ECE per class (one-vs-rest calibration error)
#     ece_pre  = _ece_per_class(probs_pre,  y_test, n_bins=n_bins, strategy=strategy)
#     ece_post = _ece_per_class(probs_post, y_test, n_bins=n_bins, strategy=strategy)

#     # Argmax stability (should be 100%)
#     argmax_same = float(np.mean(probs_pre.argmax(axis=1) == probs_post.argmax(axis=1)))

#     # 4) PRINT outcomes
#     print(f"[TS] Val NLL: {nll_before:.4f} -> {nll_after:.4f}  (Δ {nll_after - nll_before:+.4f})")
#     print(f"[TS] Test macro AUROC (invariant): {auc_macro_pre:.4f} -> {auc_macro_post:.4f}")
#     print(f"[TS] Test Brier: {brier_pre:.4f} -> {brier_post:.4f}  (Δ {brier_post - brier_pre:+.4f})")
#     print(f"[TS] Test argmax unchanged: {argmax_same*100:.2f}% of samples")
#     print("[TS] Per-class ECE (↓ better):")
#     C = probs_pre.shape[1]
#     for c in range(C):
#         e0, e1 = ece_pre.get(f"class{c}"), ece_post.get(f"class{c}")
#         e0s = "nan" if e0 is None or not np.isfinite(e0) else f"{e0:.3f}"
#         e1s = "nan" if e1 is None or not np.isfinite(e1) else f"{e1:.3f}"
#         delta = "nan" if ("nan" in (e0s, e1s)) else f"{(e1 - e0):+.3f}"
#         print(f"  class {c}: {e0s} -> {e1s}  (Δ {delta})")

#     return {
#         "val_nll_before": float(nll_before),
#         "val_nll_after":  float(nll_after),
#         "test_auc_macro_before": float(auc_macro_pre),
#         "test_auc_macro_after":  float(auc_macro_post),
#         "test_brier_before": brier_pre,
#         "test_brier_after":  brier_post,
#         "test_ece_before": ece_pre,
#         "test_ece_after":  ece_post,
#         "test_argmax_same_frac": argmax_same,
#     }


# ------------
# Temperature with modified brier
# -----------

# Utilities assumed to exist from previous context:
# - prepare_probs_targets_df(df): returns probs, y_true, logits
# - _ece_per_class(probs, y, n_bins, strategy): returns ECE per class as dict

def _nll_from_logits_np(logits: np.ndarray, y: np.ndarray) -> float:
    l = torch.from_numpy(logits).float()
    t = torch.from_numpy(y).long()
    return float(F.cross_entropy(l, t, reduction="mean").item())

def fit_temperature_on_df(df: pd.DataFrame, max_iter: int = 100, lr: float = 1e-2):
    _, y, logits = prepare_probs_targets_df(df)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    l = torch.from_numpy(logits).to(device).float()
    t = torch.from_numpy(y).to(device).long()

    logT = torch.nn.Parameter(torch.zeros(()).to(device))  # T=1
    opt = torch.optim.LBFGS([logT], lr=lr, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(l / torch.exp(logT), t)
        loss.backward()
        return loss

    opt.step(closure)

    @torch.no_grad()
    def calibrate(logits_np: np.ndarray) -> np.ndarray:
        L = torch.from_numpy(logits_np).to(device).float()
        p = F.softmax(L / torch.exp(logT), dim=1).cpu().numpy()
        return p

    nll_before = _nll_from_logits_np(logits, y)
    nll_after  = _nll_from_logits_np((logits / np.exp(logT.detach().cpu().numpy())), y)
    return calibrate, float(nll_before), float(nll_after)

def modified_brier_score(probs: np.ndarray, y: np.ndarray) -> float:
    onehot = np.eye(probs.shape[1])[y]
    mse = np.sum((probs - onehot) ** 2, axis=1)
    prevalence = np.mean(onehot, axis=0)
    uncertainty = np.sum(prevalence * (1 - prevalence))
    return float(np.mean(mse) - uncertainty)

def apply_and_report_temperature_scaling_mb(df_val, df_test, n_bins=10, strategy="quantile"):
    calibrate, nll_before, nll_after = fit_temperature_on_df(df_val)

    probs_pre, y_test, logits_test = prepare_probs_targets_df(df_test)
    probs_post = calibrate(logits_test)

    auc_macro_pre  = roc_auc_score(y_test, probs_pre,  multi_class="ovr", average="macro")
    auc_macro_post = roc_auc_score(y_test, probs_post, multi_class="ovr", average="macro")

    brier_pre  = modified_brier_score(probs_pre, y_test)
    brier_post = modified_brier_score(probs_post, y_test)

    ece_pre  = _ece_per_class(probs_pre,  y_test, n_bins=n_bins, strategy=strategy)
    ece_post = _ece_per_class(probs_post, y_test, n_bins=n_bins, strategy=strategy)

    argmax_same = float(np.mean(probs_pre.argmax(axis=1) == probs_post.argmax(axis=1)))

    print(f"[TS] Val NLL: {nll_before:.4f} -> {nll_after:.4f}  (Δ {nll_after - nll_before:+.4f})")
    print(f"[TS] Test macro AUROC (invariant): {auc_macro_pre:.4f} -> {auc_macro_post:.4f}")
    print(f"[TS] Test MODIFIED Brier: {brier_pre:.4f} -> {brier_post:.4f}  (Δ {brier_post - brier_pre:+.4f})")
    print(f"[TS] Test argmax unchanged: {argmax_same*100:.2f}% of samples")
    print("[TS] Per-class ECE (\u2193 better):")
    C = probs_pre.shape[1]
    for c in range(C):
        e0, e1 = ece_pre.get(f"class{c}"), ece_post.get(f"class{c}")
        e0s = "nan" if e0 is None or not np.isfinite(e0) else f"{e0:.3f}"
        e1s = "nan" if e1 is None or not np.isfinite(e1) else f"{e1:.3f}"
        delta = "nan" if ("nan" in (e0s, e1s)) else f"{(e1 - e0):+.3f}"
        print(f"  class {c}: {e0s} -> {e1s}  (Δ {delta})")

    return {
        "val_nll_before": nll_before,
        "val_nll_after":  nll_after,
        "test_auc_macro_before": auc_macro_pre,
        "test_auc_macro_after":  auc_macro_post,
        "test_brier_before": brier_pre,
        "test_brier_after":  brier_post,
        "test_ece_before": ece_pre,
        "test_ece_after":  ece_post,
        "test_argmax_same_frac": argmax_same,
    }

# ---------
# ACCURACY
# ---------

import numpy as np, pandas as pd, ast
from sklearn.metrics import confusion_matrix

def _wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson 95% CI for a proportion k/n. Returns (p_hat, (lo, hi))."""
    if n <= 0:
        return np.nan, (np.nan, np.nan)
    phat = k / n
    denom = 1.0 + (z**2)/n
    center = (phat + (z**2)/(2*n)) / denom
    half   = z*np.sqrt((phat*(1-phat) + (z**2)/(4*n))/n) / denom
    return phat, (max(0.0, center - half), min(1.0, center + half))

def _coerce_logits_column(col, n_classes=None):
    """Accepts list/np/torch or JSON-like strings; returns stacked (N,C) float array."""
    def to_arr(v):
        if isinstance(v, str):
            v = ast.literal_eval(v)
        try:
            import torch
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
        except Exception:
            pass
        return np.asarray(v, dtype=float).reshape(-1)
    arrs = [to_arr(v) for v in col]
    if n_classes is not None:
        for a in arrs:
            if a.size != n_classes:
                raise ValueError(f"Found logits of size {a.size}, expected {n_classes}.")
    return np.stack(arrs, axis=0)

def accuracy_panel_from_df(df: pd.DataFrame, logits_col="cls_logits", label_col="cls_true"):
    """
    Returns:
      {
        'overall': {'acc': float, 'n': int, 'ci': (lo, hi)},
        'per_class': {'class0': {...}, ...},   # per-class 'accuracy' = recall_c
        'balanced_accuracy': float,
        'confusion_matrix': [[...], ...],
      }
    """
    logits = _coerce_logits_column(df[logits_col])
    y_true = pd.to_numeric(df[label_col], errors="coerce").to_numpy().astype(int)
    mask = ~np.isnan(y_true)
    logits, y_true = logits[mask], y_true[mask]
    C = logits.shape[1]

    # Argmax predictions (softmax not needed for argmax)
    y_pred = logits.argmax(axis=1)

    # Confusion matrix (rows=true, cols=pred)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(C))
    n_total = int(cm.sum())
    n_correct = int(np.trace(cm))

    # Overall accuracy + 95% CI (Wilson)
    overall_acc, overall_ci = _wilson_ci(n_correct, n_total)

    # Per-class "accuracy within class" = recall (TP_c / N_c) + 95% CI
    per_class = {}
    recalls = []
    for c in range(C):
        n_c = int(cm[c, :].sum())
        tp_c = int(cm[c, c])
        acc_c, ci_c = _wilson_ci(tp_c, n_c)
        per_class[f"class{c}"] = {"acc": acc_c, "n": n_c, "ci": ci_c}
        if n_c > 0:
            recalls.append(acc_c)

    # Balanced accuracy = mean recall over present classes
    bal_acc = float(np.mean(recalls)) if recalls else np.nan

    return {
        "overall": {"acc": overall_acc, "n": n_total, "ci": overall_ci},
        "per_class": per_class,
        "balanced_accuracy": bal_acc,
        "confusion_matrix": cm.tolist(),
    }

# =======
# Binary with CIs
# =======
def plot_binary_roc_by_bin(
    df,
    subgroup_col='time_diff_bins',
    logits_col='cls_logits',
    label_col='cls_true',
    positive_class=2,
    negative_class=1,
    save_path=None,
    title=None,
    figsize=(7, 7),
    n_boot=2000,
    ci=0.95,
    random_state=42,
):
    """
    Plot one ROC curve per bin for a binary task: positive_class vs negative_class
    (controls / other classes excluded).

    Also computes bootstrap confidence intervals for AUC within each bin.
    """

    logits = np.stack(df[logits_col].values)
    probs = _softmax_np(logits, axis=1)

    rng = np.random.default_rng(random_state)

    def bootstrap_auc_ci(y, p, n_boot=2000, ci=0.95):
        aucs = []
        n = len(y)

        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            y_b = y[idx]
            p_b = p[idx]

            # bootstrap sample must contain both classes
            if len(np.unique(y_b)) < 2:
                continue

            aucs.append(roc_auc_score(y_b, p_b))

        if len(aucs) == 0:
            return np.nan, np.nan

        alpha = 1 - ci
        lo = np.percentile(aucs, 100 * alpha / 2)
        hi = np.percentile(aucs, 100 * (1 - alpha / 2))
        return lo, hi

    def sort_key(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return v

    bin_values = sorted(df[subgroup_col].dropna().unique(), key=sort_key)
    colors = sns.color_palette('colorblind', n_colors=len(bin_values))

    bin_label = {
        1: '0 - 6 h  ',
        2: '6 - 12 h ',
        3: '12 - 24 h'
    }

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.6)

    results = []

    for color, bin_val in zip(colors, bin_values):
        mask_bin = (df[subgroup_col] == bin_val).fillna(False).to_numpy()

        y_bin_all = df.loc[mask_bin, label_col].to_numpy().astype(int)
        p_bin_all = probs[mask_bin, positive_class]

        # keep only positive_class and negative_class
        keep = np.isin(y_bin_all, [positive_class, negative_class])

        y = (y_bin_all[keep] == positive_class).astype(int)
        p = p_bin_all[keep]

        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)

        label_prefix = bin_label.get(bin_val, str(bin_val))

        if n_pos < 2 or n_neg < 2:
            ax.plot(
                [], [],
                color=color,
                label=f'{label_prefix}: insufficient data '
                      f'(n_pos={n_pos}, n_neg={n_neg})'
            )

            results.append({
                subgroup_col: bin_val,
                "n": len(y),
                "n_pos": n_pos,
                "n_neg": n_neg,
                "auc": np.nan,
                "auc_ci_low": np.nan,
                "auc_ci_high": np.nan,
            })

            continue

        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        auc_lo, auc_hi = bootstrap_auc_ci(y, p, n_boot=n_boot, ci=ci)

        ax.plot(
            fpr,
            tpr,
            color=color,
            lw=2,
            label=(
                f'{label_prefix} '
                f'(n_pos={n_pos}, n_neg={n_neg}, '
                f'AUC={auc:.2f} [{auc_lo:.2f}, {auc_hi:.2f}])'
            )
        )

        results.append({
            subgroup_col: bin_val,
            "n": len(y),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "auc": auc,
            "auc_ci_low": auc_lo,
            "auc_ci_high": auc_hi,
        })

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

    default_title = (
        f'ROC by {subgroup_col}: class {positive_class} '
        f'vs class {negative_class}'
    )
    if title: ax.set_title(title)

    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()

    results_df = pd.DataFrame(results)

    return fig, ax, results_df