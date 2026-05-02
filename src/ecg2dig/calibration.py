# --- calibration_eval.py ---
from __future__ import annotations
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
import numpy as np
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    f1_score, balanced_accuracy_score, matthews_corrcoef,
    confusion_matrix, cohen_kappa_score
)
from sklearn.calibration import calibration_curve

try:
    from sklearn.metrics import roc_auc_score, log_loss
    HAS_SK = True
except Exception:
    HAS_SK = False
    
# --------------
# Temperature calibrator adapted from paper:
# Guo C, Pleiss G, Sun Y, Weinberger K. Q.On Calibration of Modern Neural Networks. 2017.
# --------------

class TemperatureScaler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temp = torch.nn.Parameter(torch.zeros(()))  # log T; init T=1

    def forward(self, logits):
        # Clamp log T to [-7, 7] → T in [~9e-4, ~1096]; keeps optimization stable
        logT = torch.clamp(self.log_temp, -7.0, 7.0)
        T = torch.exp(logT) + 1e-6
        return logits / T

def fit_temperature(df_to_cal, col_logits='cls_logits', col_true='cls_true',
                    task_type='multiclass', max_iter=200, lr=0.05, seed=42):
    """
    Fit temperature scaling on a validation dataframe.
    df_val[col_logits]: sequence of length-K logit vectors
    df_val[col_true]: integer class ids (0..K-1) or {0,1} for binary
    """
    df = df_to_cal.copy()
    Z = np.vstack(df[col_logits].to_numpy()).astype(np.float64)   # [N, K] or [N, 1]
    y = df[col_true].to_numpy()

    logits = torch.from_numpy(Z)
    
    torch.manual_seed(seed)
    #scaler = TemperatureScaler().to(logits.device, logits.dtype)
    scaler = TemperatureScaler().to(dtype=torch.float64, device=logits.device)

    opt = torch.optim.LBFGS(
        scaler.parameters(), lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe"
    )

    # choose loss according to task_type and logits shape
    if task_type == 'binary':
        if Z.shape[1] == 1:  # single-logit binary
            y_t = torch.from_numpy(y).to(logits.device, logits.dtype).view(-1, 1)
            loss_fn = torch.nn.BCEWithLogitsLoss()
        elif Z.shape[1] == 2:  # two-logit binary; treat as multiclass CE
            y_t = torch.from_numpy(y).to(logits.device).long().view(-1)
            loss_fn = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError("binary task expects logits with 1 or 2 columns.")
    else:
        y_t = torch.from_numpy(y).to(logits.device).long().view(-1)
        loss_fn = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = loss_fn(scaler(logits), y_t)
        loss.backward()
        return loss

    # Single LBFGS call performs up to max_iter internal iterations
    opt.step(closure)
    return scaler

# Add dataframe calibrated probs column for Temperature scaling

def add_calibrated_temp_probs(df_to_cal, calibrator, col_logits='cls_logits'):
    """
    Adds both raw (softmax) and calibrated probabilities to df_test.
    Returns a new DataFrame (df_test_cal).

    - Raw probs: softmax(logits)
    - Calibrated (Temperature): softmax(calibrator(logits))
    - Calibrated (Dirichlet, if used): calibrator.transform_logits(logits)  # returns probs
    """
    df = df_to_cal.copy() 
    Z = np.vstack(df[col_logits].to_numpy()).astype(np.float64) # [N, K]

    # --- raw softmax probabilities (uncalibrated) ---
    Zs = Z - Z.max(axis=1, keepdims=True)
    P_raw = np.exp(Zs) / np.exp(Zs).sum(axis=1, keepdims=True)

    # --- calibrated probabilities ---
    calibrator.eval()
    
    
    
#     with torch.no_grad():
#     if hasattr(calibrator, "transform_logits"):
#         P_cal = calibrator.transform_logits(Z).astype(np.float64)
#     else:
#         logits_t = torch.from_numpy(Z).to(device=next(calibrator.parameters()).device,
#                                           dtype=torch.float64)        # <- 64-bit input
#         scaled = calibrator(logits_t)                                  # logits / T (float64)
#         P_cal = torch.softmax(scaled, dim=1).to(torch.float64).cpu().numpy()  # <- 64-bit out
    
    
    
    with torch.no_grad():
        if hasattr(calibrator, "transform_logits"):
            P_cal = calibrator.transform_logits(Z).astype(np.float64)
        else:
            # Temperature scaling: softmax of scaled logits
            dtype = next(calibrator.parameters()).dtype
            device = next(calibrator.parameters()).device
            logits_t = torch.from_numpy(Z).to(device=device, dtype=dtype)
            scaled = calibrator(logits_t)                          # logits / T
            P_cal = torch.softmax(scaled, dim=1).cpu().numpy()     # calibrated probs

    df['cls_p_raw'] = [row for row in P_raw]
    df['cls_p_cal'] = [row for row in P_cal]
    return df


# -----------
# Dirichlet Calibrator
# -----------

class DirichletCalibrator(torch.nn.Module):
    """
    p_cal = softmax( W @ log(p) + b ), where p = softmax(logits)
    Fit W,b on validation by minimizing NLL (cross-entropy).
    """
    def __init__(self, K, eps=1e-12):
        super().__init__()
        self.K = K
        self.eps = eps
        self.W = torch.nn.Parameter(torch.eye(K, dtype=torch.float64))
        self.b = torch.nn.Parameter(torch.zeros(K, dtype=torch.float64))

    @torch.no_grad()
    def transform_logits(self, logits_np: np.ndarray) -> np.ndarray:
        Z = torch.from_numpy(logits_np.astype(np.float64))
        p = torch.softmax(Z, dim=1).clamp_(min=self.eps)
        z = (torch.log(p) @ self.W.T) + self.b
        return torch.softmax(z, dim=1).cpu().numpy()

def fit_dirichlet_on_validation(df_to_cal, col_logits='cls_logits', col_true='cls_true',
                                lr=5e-2, max_iter=2000, weight_decay=0.0, seed=42):
    
    df = df_to_cal.copy()
    Z = np.vstack(df[col_logits].to_numpy()).astype(np.float64)  # [N,K]
    y = df[col_true].to_numpy().astype(int)
    K = Z.shape[1]

    torch.manual_seed(seed)
    cal = DirichletCalibrator(K)
    opt = torch.optim.Adam(cal.parameters(), lr=lr, weight_decay=weight_decay)

    Zt = torch.from_numpy(Z).to(torch.float64)
    yt = torch.from_numpy(y).to(torch.long)

    cal.train()
    for _ in range(max_iter):
        opt.zero_grad()
        p = torch.softmax(Zt, dim=1).clamp(min=cal.eps)
        z = (torch.log(p) @ cal.W.T) + cal.b
        loss = F.cross_entropy(z.to(torch.float64), yt)  # NLL on validation
        loss.backward()
        opt.step()

    cal.eval()
    return cal

# ------------------
# Add column for the calibrated probailtites
# ------------------
def add_calibrated_probs(df_to_cal, calibrator, col_logits='cls_logits'):
    """
    Adds both raw (softmax) and calibrated probabilities to df_test.
    Returns a new DataFrame (df_test_cal).
    """
    df = df_to_cal.copy()
    Z = np.vstack(df[col_logits].to_numpy()).astype(np.float64)

    # --- raw softmax probabilities (uncalibrated) ---
    Zs = Z - Z.max(axis=1, keepdims=True)
    P_raw = np.exp(Zs) / np.exp(Zs).sum(axis=1, keepdims=True)

    # --- Dirichlet-calibrated probabilities ---
    P_cal = calibrator.transform_logits(Z)

    # add both columns
    df['cls_p_raw'] = list(P_raw)
    df['cls_p_cal'] = list(P_cal)


    return df

    
# ----------------------------
# 1) Collect logits and targets
# ----------------------------
@torch.no_grad()
def collect_logits_targets(model, loader, device):
    model.eval()
    all_logits, all_targets = [], []
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        cls_out, attn_cls, _, _, _, _ = model(x)  # your tuple
        all_logits.append(cls_out.detach().float().cpu())
        all_targets.append(y.detach().cpu())
    logits = torch.cat(all_logits, dim=0)         # [N, K]
    targets = torch.cat(all_targets, dim=0)       # [N] or [N, K]
    return logits, targets

# ----------------------------
# 2) Helpers for task typing
# ----------------------------
def infer_task_type(logits, targets):
    """
    Returns ('binary', K) or ('softmax2', 2)
    """
    if logits.ndim == 1:
        return 'binary', 1
    if logits.shape[1] == 1:
        return 'binary', 1
    if logits.shape[1] == 2:
        return 'softmax2', 2
    raise ValueError(f"Unsupported CLS shape: {tuple(logits.shape)}")

def logits_to_probs(logits, task_type):
    if task_type == 'binary':
        return torch.sigmoid(logits)              # [N,1]
    elif task_type == 'softmax2':
        return torch.softmax(logits, dim=1)[:, 1:2]  # [N,1] = P(class=1)
    else:
        raise ValueError(task_type)

def ce_loss(logits, targets, task_type):
    if task_type == 'binary':
        y = targets.float().view(-1, 1)
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    else:  # softmax2
        y = targets.view(-1).long()
        return torch.nn.functional.cross_entropy(logits, y)

def compute_auc(probs, targets, task_type):
    if not HAS_SK: 
        return None
    p = probs.detach().cpu().numpy().reshape(-1)
    if task_type == 'binary':
        y = targets.view(-1).cpu().numpy().astype(int)
    else:
        y = targets.view(-1).cpu().numpy().astype(int)
    # If only one class present, roc_auc_score will error; guard:
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))

# ---------------------------------
# 3) ECE + reliability visualization
# ---------------------------------
def calibration_curve_and_ece(probs, targets, n_bins=15):
    p = probs.detach().cpu().numpy().reshape(-1)
    y = targets.view(-1).cpu().numpy().astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    inds = np.digitize(p, bins) - 1
    bin_acc, bin_conf, bin_count = [], [], []
    ece = 0.0
    for b in range(n_bins):
        m = inds == b
        if not np.any(m):
            bin_acc.append(np.nan); bin_conf.append(np.nan); bin_count.append(0)
            continue
        conf = p[m].mean()
        acc  = y[m].mean()
        n    = m.sum()
        bin_acc.append(acc); bin_conf.append(conf); bin_count.append(n)
        ece += (n / len(p)) * abs(acc - conf)
    return np.array(bin_conf), np.array(bin_acc), np.array(bin_count), float(ece)

def plot_reliability(bin_conf, bin_acc, title="Reliability diagram"):
    mask = ~np.isnan(bin_acc)
    plt.figure()
    plt.plot([0,1],[0,1],'--',linewidth=1)
    plt.plot(bin_conf[mask], bin_acc[mask], marker='o')
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical accuracy")
    plt.title(title)
    plt.grid(True, linestyle=':')
    plt.show()

# ----------------------------
# 4) Brier score
# ----------------------------
def brier_score(probs, targets):
    p = probs.detach()
    y = targets.float().view_as(p)
    return torch.mean((p - y)**2).item()

# ----------------------------
# 6) End-to-end evaluate
# ----------------------------
def evaluate_calibration(model, val_loader, device="cuda"):
    logits, targets = collect_logits_targets(model, val_loader, device)

    # --- mask invalid targets (NaN or inf) ---
    mask = torch.isfinite(targets.view(-1))
    logits, targets = logits[mask], targets[mask]

    if logits.numel() == 0:
        raise ValueError("No valid samples after removing NaNs.")

    # Infer task type
    task_type, _ = infer_task_type(logits, targets)

    # --- Raw metrics ---
    ce = ce_loss(logits, targets, task_type).item()
    probs = logits_to_probs(logits, task_type)
    auc  = compute_auc(probs, targets, task_type)
    bin_conf, bin_acc, bin_count, ece = calibration_curve_and_ece(probs, targets, n_bins=15)
    brier = brier_score(probs, targets)

    print("== Raw (pre-calibration) ==")
    print(f"CE:    {ce:.4f}")
    print(f"Brier: {brier:.4f}")
    if auc is not None:
        print(f"AUC:   {auc:.4f}")
    else:
        print("AUC:   (undefined: single-class in val set)")
    plot_reliability(bin_conf, bin_acc, title="Reliability (raw)")
    print(f"ECE:   {ece:.4f}")

    # --- Temperature scaling ---
    scaler = fit_temperature(logits, targets, task_type)
    with torch.no_grad():
        logits_cal = scaler(logits)
        probs_cal  = logits_to_probs(logits_cal, task_type)
        ce_cal     = ce_loss(logits_cal, targets, task_type).item()
        auc_cal    = compute_auc(probs_cal, targets, task_type)
        bin_conf_c, bin_acc_c, _, ece_c = calibration_curve_and_ece(probs_cal, targets, n_bins=15)
        brier_cal  = brier_score(probs_cal, targets)

    print("\n== After temperature scaling ==")
    print(f"T:     {float(torch.exp(scaler.log_temp)):.4f}")
    print(f"CE:    {ce_cal:.4f}  (Δ {ce_cal - ce:+.4f})")
    print(f"Brier: {brier_cal:.4f}  (Δ {brier_cal - brier:+.4f})")
    if auc_cal is not None:
        print(f"AUC:   {auc_cal:.4f}  (Δ {0.0 if auc is None else auc_cal - auc:+.4f})")
    plot_reliability(bin_conf_c, bin_acc_c, title="Reliability (temp-scaled)")
    print(f"ECE:   {ece_c:.4f}  (Δ {ece_c - ece:+.4f})")

# ----------------------------
# Usage:
# ----------------------------
# model = torch.load("/path/to/model_best_full_model.pt", map_location="cuda" if torch.cuda.is_available() else "cpu")
# model.eval()
# evaluate_calibration(model, val_loader, device="cuda" if torch.cuda.is_available() else "cpu")

# -
# - new ones

# ===============================
# 1) MODEL → LOADER → LOGITS/Y
# ===============================

def _extract_cls_logits(model_out):
    """
    Flexible extractor: works if model returns:
      - a Tensor of shape (N, 3), or
      - a dict with key 'cls_logits' (Tensor (N,3)), or
      - a tuple/list where first elem are cls logits
    Edit if your API differs.
    """
    if isinstance(model_out, dict) and "cls_logits" in model_out:
        return model_out["cls_logits"]
    if isinstance(model_out, (tuple, list)) and len(model_out) > 0:
        out0 = model_out[0]
        if torch.is_tensor(out0) and out0.dim() == 2:
            return out0
        if isinstance(out0, dict) and "cls_logits" in out0:
            return out0["cls_logits"]
    if torch.is_tensor(model_out) and model_out.dim() == 2:
        return model_out
    raise ValueError("Could not locate CLS logits in model output.")

@torch.no_grad()
def collect_logits_and_targets(model, loader, device="cuda", use_amp=True, cls_target_key="high_digoxin"):
    """
    Returns:
        logits: (N, 3) float32 numpy
        y     : (N,)   int64 numpy in {0,1,2}
    Notes:
      - Assumes batch is (ecg, *extras, target_dict/target_tensor)
      - If targets are a dict, it tries cls_target_key; else assumes tensor.
    """
    model.eval()
    logits_lst, targets_lst = [], []
    amp_ctx = torch.cuda.amp.autocast if (use_amp and device.startswith("cuda")) else nullcontext

    for batch in loader:
        # Try common batch structures: (x, y) or (x, *meta, y)
        if isinstance(batch, (list, tuple)):
            x = batch[0]
            tgt = batch[-1]
        else:
            # dict-style: {'ecg':..., 'targets':...}
            x = batch["ecg"]
            tgt = batch["targets"]

        x = x.to(device, non_blocking=True)

        # Targets
        if isinstance(tgt, dict):
            y = tgt.get(cls_target_key, None)
            if y is None:
                raise KeyError(f"targets dict missing '{cls_target_key}'")
        else:
            y = tgt
        y = y.to(device)

        with amp_ctx():
            out = model(x)
            cls_logits = _extract_cls_logits(out)

        logits_lst.append(cls_logits.detach().float().cpu())
        targets_lst.append(y.detach().long().cpu())

    logits = torch.cat(logits_lst, dim=0).numpy()
    y = torch.cat(targets_lst, dim=0).numpy()

    # Drop NaN labels if present
    mask = ~np.isnan(y)
    if mask.dtype != bool:  # integer labels -> no NaNs
        mask = np.ones_like(y, dtype=bool)
    logits = logits[mask]
    y = y[mask].astype(int)

    return logits, y

def _softmax_np(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    ex = np.exp(logits)
    return ex / ex.sum(axis=1, keepdims=True)

# ==========================================
# 2) METRICS (multiclass, ordinal-friendly)
# ==========================================

import time
import torch
import numpy as np
from contextlib import nullcontext
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, balanced_accuracy_score, matthews_corrcoef,
    confusion_matrix, cohen_kappa_score
)

# -------- existing helper ----------
def _softmax_np(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    ex = np.exp(logits)
    return ex / ex.sum(axis=1, keepdims=True)

def _extract_cls_and_hr(model_out):
    if isinstance(model_out, dict):
        return model_out["cls_logits"], model_out.get("hr_out", None)
    if isinstance(model_out, (tuple, list)) and len(model_out) >= 2:
        return model_out[0], model_out[1]
    if torch.is_tensor(model_out) and model_out.dim() == 2:
        return model_out, None
    raise ValueError("Could not parse (cls_logits, hr_out) from model output.")

# -------- NEW: verbose collector ----------
@torch.no_grad()
def collect_cls_hr_from_loader(
    model, loader, device="cuda", use_amp=True,
    cls_target_key="cls_true", hr_target_key="hr",
    verbose=False, print_every=50, max_batches=None
):
    """
    Returns:
        logits_np: (N,3)  y_cls_np: (N,)  hr_pred: (N,) or None  hr_true: (N,) or None
    """
    model.eval()
    amp_ctx = torch.cuda.amp.autocast if (use_amp and device.startswith("cuda")) else nullcontext

    logits_list, y_cls_list = [], []
    hr_pred_list, hr_true_list = [], []

    total_batches = len(loader) if hasattr(loader, "__len__") else None
    seen = 0
    t0 = time.time()
    if verbose:
        tb = f"/{total_batches}" if total_batches is not None else ""
        print(f"[collect] Starting… total batches{tb}")

    for b, batch in enumerate(loader, 1):
        # unpack
        if isinstance(batch, (tuple, list)):
            x = batch[0]; targets = batch[-1]
        else:
            x = batch["ecg"]; targets = batch["targets"]

        x = x.to(device, non_blocking=True)

        # targets
        if isinstance(targets, dict):
            y_cls = targets[cls_target_key]
            y_hr  = targets.get(hr_target_key, None)
        else:
            y_cls, y_hr = targets, None
        y_cls = y_cls.to(device)

        with amp_ctx():
            out = model(x)
            cls_logits, hr_out = _extract_cls_and_hr(out)

        logits_list.append(cls_logits.detach().float().cpu())
        y_cls_list.append(y_cls.detach().long().cpu())
        n = x.shape[0]
        seen += n

        if hr_out is not None:
            hr_pred_list.append(hr_out.detach().float().cpu())
        if y_hr is not None:
            hr_true_list.append(y_hr.detach().float().cpu() if torch.is_tensor(y_hr)
                                else torch.tensor(y_hr, dtype=torch.float32))

        if verbose and (b == 1 or b % print_every == 0):
            if device.startswith("cuda"):
                torch.cuda.synchronize()
                mem = torch.cuda.memory_allocated() / 1e9
                rsv = torch.cuda.memory_reserved() / 1e9
                mem_str = f" | GPU mem alloc/resv: {mem:.2f}/{rsv:.2f} GB"
            else:
                mem_str = ""
            elapsed = time.time() - t0
            tb = f"/{total_batches}" if total_batches is not None else ""
            print(f"[collect] batch {b}{tb} | +{n} samples | total {seen} | {elapsed:.1f}s{mem_str}")

        if max_batches is not None and b >= max_batches:
            if verbose:
                print(f"[collect] Early stop at max_batches={max_batches}")
            break

    logits_np = torch.cat(logits_list, dim=0).numpy()
    y_cls_np  = torch.cat(y_cls_list, dim=0).numpy()
    if np.issubdtype(y_cls_np.dtype, np.floating):
        mask = ~np.isnan(y_cls_np)
        logits_np = logits_np[mask]; y_cls_np = y_cls_np[mask]
    y_cls_np = y_cls_np.astype(int)

    hr_pred = (torch.cat(hr_pred_list, dim=0).squeeze(-1).numpy()
               if hr_pred_list else None)
    hr_true = (torch.cat(hr_true_list, dim=0).squeeze(-1).numpy()
               if hr_true_list else None)

    if verbose:
        print(f"[collect] Done. logits{tuple(logits_np.shape)}, y{tuple(y_cls_np.shape)}, "
              f"hr_pred={'None' if hr_pred is None else hr_pred.shape}, "
              f"elapsed={time.time()-t0:.1f}s")
    return logits_np, y_cls_np, hr_pred, hr_true


def evaluate_multiclass_metrics_from_loader(
    model, loader, device="cuda", use_amp=True,
    verbose=False, print_every=50, max_batches=None
):
    t0 = time.time()
    if verbose: print("[eval] Collecting logits/targets…")
    logits, y, _, _ = collect_cls_hr_from_loader(
        model, loader, device, use_amp,
        verbose=verbose, print_every=print_every, max_batches=max_batches
    )
    if verbose: print(f"[eval] Collected {len(y)} samples in {time.time()-t0:.1f}s. Computing probabilities…")
    probs = _softmax_np(logits)
    preds = probs.argmax(axis=1)

    # AUCs
    if verbose: print("[eval] AUROC macro/weighted…")
    try:
        auc_macro = roc_auc_score(y, probs, multi_class="ovr", average="macro")
        auc_weighted = roc_auc_score(y, probs, multi_class="ovr", average="weighted")
    except ValueError:
        auc_macro = np.nan; auc_weighted = np.nan

    # Per-class
    per_class = {}
    C = probs.shape[1]
    for c in range(C):
        y_bin = (y == c).astype(int)
        if verbose:
            pos = y_bin.sum()
            print(f"[eval] class {c}: positives={int(pos)}, computing ROC-AUC/AP…")
        if len(np.unique(y_bin)) > 1:
            auc = roc_auc_score(y_bin, probs[:, c])
            ap  = average_precision_score(y_bin, probs[:, c])
        else:
            auc, ap = np.nan, np.nan
        per_class[f"class{c}"] = {"AUC": auc, "AP": ap}

    if verbose: print("[eval] Confusion matrix, F1/MCC/QWK…")
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
    if verbose: print(f"[eval] Done in {time.time()-t0:.1f}s.")
    return metrics, probs, y

# ==========================================
# 3) DECISION CURVE (class 2 vs {0,1})
# ==========================================

def plot_decision_curve_analysis_from_loader(model, loader, device="cuda", use_amp=True, positive_class=2):
    _, probs, y = _ensure_probs_targets(model, loader, device, use_amp)
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

# ==========================================
# 4) ROC (3 classes in ONE figure)
# ==========================================

def plot_multiclass_roc_from_loader(model, loader, device="cuda", use_amp=True, class_names=("0","1","2")):
    _, probs, y = _ensure_probs_targets(model, loader, device, use_amp)
    plt.figure()
    for c in range(probs.shape[1]):
        y_bin = (y == c).astype(int)
        if len(np.unique(y_bin)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_bin, probs[:, c])
        auc = roc_auc_score(y_bin, probs[:, c])
        plt.plot(fpr, tpr, label=f"{class_names[c]} (AUC={auc:.3f})")
    plt.plot([0,1],[0,1],"--", alpha=0.6)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC – One-vs-Rest (all classes)")
    plt.legend()
    plt.show()

# ==========================================
# 5) PR (3 classes in ONE figure)
# ==========================================

def plot_multiclass_pr_from_loader(model, loader, device="cuda", use_amp=True, class_names=("0","1","2")):
    _, probs, y = _ensure_probs_targets(model, loader, device, use_amp)
    plt.figure()
    for c in range(probs.shape[1]):
        y_bin = (y == c).astype(int)
        if len(np.unique(y_bin)) < 2:
            continue
        prec, rec, _ = precision_recall_curve(y_bin, probs[:, c])
        ap = average_precision_score(y_bin, probs[:, c])
        plt.plot(rec, prec, label=f"{class_names[c]} (AP={ap:.3f})")
        # prevalence reference
        plt.hlines(y_bin.mean(), xmin=0, xmax=1, linestyles="dotted", alpha=0.3)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall – One-vs-Rest (all classes)")
    plt.legend()
    plt.xlim(0,1); plt.ylim(0,1)
    plt.show()

# ==========================================
# 6) CALIBRATION (ECE, Brier, reliability)
# ==========================================

def _ece_per_class(probs, y, n_bins=10, strategy="quantile"):
    eces = {}
    for c in range(probs.shape[1]):
        y_bin = (y == c).astype(int)
        if len(np.unique(y_bin)) < 2:
            eces[f"class{c}"] = np.nan
            continue
        # calibration_curve returns avg predicted prob per bin (prob_pred) and empirical freq (prob_true)
        prob_true, prob_pred = calibration_curve(y_bin, probs[:, c], n_bins=n_bins, strategy=strategy)
        # ECE = sum_k (|bin_k|/N) * |prob_true_k - prob_pred_k|
        # Approximate bin weights as uniform over returned points (better: compute explicit bin sizes)
        # We'll recompute explicit binning to get weights:
        # build bins by predicted prob quantiles if strategy='quantile'
        p = probs[:, c]
        if strategy == "uniform":
            bins = np.linspace(0, 1, n_bins+1)
        else:
            bins = np.quantile(p, np.linspace(0, 1, n_bins+1))
            bins[0], bins[-1] = 0.0, 1.0
        inds = np.digitize(p, bins) - 1
        inds = np.clip(inds, 0, n_bins-1)
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

def evaluate_calibration_from_loader(model, loader, device="cuda", use_amp=True, n_bins=10, strategy="quantile", plot=True):
    logits, y = collect_logits_and_targets(model, loader, device, use_amp)
    probs = _softmax_np(logits)

    # Multiclass Brier
    onehot = np.zeros_like(probs); onehot[np.arange(len(y)), y] = 1
    brier = np.mean(np.sum((probs - onehot)**2, axis=1))

    # Per-class reliability curves + ECE
    curves = {}
    for c in range(probs.shape[1]):
        y_bin = (y == c).astype(int)
        if len(np.unique(y_bin)) < 2:
            curves[f"class{c}"] = None
            continue
        prob_true, prob_pred = calibration_curve(y_bin, probs[:, c], n_bins=n_bins, strategy=strategy)
        curves[f"class{c}"] = {"prob_true": prob_true, "prob_pred": prob_pred}

    eces = _ece_per_class(probs, y, n_bins=n_bins, strategy=strategy)

    if plot:
        for c in range(probs.shape[1]):
            data = curves.get(f"class{c}")
            if data is None: 
                continue
            plt.figure()
            plt.plot(data["prob_pred"], data["prob_true"], marker="o", label=f"class {c}")
            plt.plot([0,1],[0,1], "--", alpha=0.6)
            plt.xlabel("Predicted probability")
            plt.ylabel("Empirical frequency")
            plt.title(f"Reliability – class {c} (ECE={eces[f'class{c}']:.3f})")
            plt.legend()
            plt.xlim(0,1); plt.ylim(0,1)
            plt.show()

    return {"brier": brier, "ece_per_class": eces, "curves": curves, "probs": probs, "y": y}

# ===============================================================
# 7a) (Optional) Temperature scaling using the model and a loader
# ===============================================================

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.logT = nn.Parameter(torch.zeros(()))  # T=1.0 initially
    
    def forward(self, logits):
        T = torch.exp(self.logT)
        return logits / T
    
    @property
    def temperature(self):
        return torch.exp(self.logT).item()
    
@torch.no_grad()
def _nll_from_logits(logits, y):
    # logits: (N,C), y:(N,)
    return F.cross_entropy(logits, y, reduction="mean").item()

def fit_temperature_on_loader(model, loader, device="cuda", use_amp=True, max_iter=100, lr=1e-2):
    """
    Fits a single temperature on logits to minimize NLL on given loader.
    Returns a callable: lambda logits_np: calibrated_probs_np
    """
    logits_np, y_np = collect_logits_and_targets(model, loader, device, use_amp)
    logits = torch.from_numpy(logits_np).to(device)
    y = torch.from_numpy(y_np).to(device)

    scaler = TemperatureScaler().to(device)
    opt = torch.optim.LBFGS(scaler.parameters(), lr=lr, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(scaler(logits), y)
        loss.backward()
        return loss

    opt.step(closure)

    @torch.no_grad()
    def calibrate(logits_np_in):
        l = torch.from_numpy(logits_np_in).to(device)
        l_cal = scaler(l)
        p = F.softmax(l_cal, dim=1).cpu().numpy()
        return p

    before = _nll_from_logits(logits, y)
    after  = _nll_from_logits(scaler(logits), y)
    return calibrate, float(before), float(after)

# ===============================
# Helper to avoid recomputation
# ===============================
def _ensure_probs_targets(model, loader, device, use_amp):
    logits, y = collect_logits_and_targets(model, loader, device, use_amp)
    probs = _softmax_np(logits)
    return logits, probs, y

# =======================================================
# 7b) (Optional) Temperature scaling using the logits df
# =======================================================

def fit_temperature_on_dataframe(df_val, device="cpu", max_iter=100, lr=1e-2):
    """
    Fits temperature on validation dataframe.
    Returns a callable for calibration and the optimal temperature.
    """
    # Extract logits and labels from dataframe
    logits_np = np.vstack(df_val['cls_logits'].values)
    y_np = df_val['cls_true'].values.astype(np.int64)
    
    # Convert to tensors
    logits = torch.from_numpy(logits_np).float().to(device)
    y = torch.from_numpy(y_np).long().to(device)
    
    # Initialize and optimize temperature scaler
    scaler = TemperatureScaler().to(device)
    opt = torch.optim.LBFGS(scaler.parameters(), lr=lr, max_iter=max_iter)
    
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(scaler(logits), y)
        loss.backward()
        return loss
    
    # Calculate NLL before optimization
    with torch.no_grad():
        nll_before = F.cross_entropy(logits, y).item()
    
    # Optimize
    opt.step(closure)
    
    # Calculate NLL after optimization
    with torch.no_grad():
        nll_after = F.cross_entropy(scaler(logits), y).item()
    
    # Create calibration function
    @torch.no_grad()
    def calibrate(logits_np_in):
        """Apply temperature scaling and return calibrated probabilities."""
        l = torch.from_numpy(logits_np_in).float().to(device)
        l_cal = scaler(l)
        p = F.softmax(l_cal, dim=1).cpu().numpy()
        return p
    
    print(f"Temperature optimization complete:")
    print(f"  Optimal T = {scaler.temperature:.3f}")
    print(f"  NLL before: {nll_before:.4f}")
    print(f"  NLL after:  {nll_after:.4f}")
    
    return calibrate, scaler.temperature

def apply_calibration_to_dataframe(df, calibrate_fn):
    """Apply calibration function to dataframe and add probability columns."""
    
    # Get calibrated probabilities
    logits = np.vstack(df['cls_logits'].values)
    probs_cal = calibrate_fn(logits)
    
    # Add columns
    df['prob_0_cal'] = probs_cal[:, 0]
    df['prob_1_cal'] = probs_cal[:, 1]
    df['prob_2_cal'] = probs_cal[:, 2]
    df['max_prob_cal'] = probs_cal.max(axis=1)
    df['pred_class_cal'] = probs_cal.argmax(axis=1)
    
    # Add confidence levels
    conditions = [
        df['max_prob_cal'] < 0.3,
        df['max_prob_cal'] < 0.7,
        df['max_prob_cal'] < 0.9,
        df['max_prob_cal'] >= 0.9
    ]
    choices = ['very_low', 'uncertain', 'confident', 'high_confidence']
    df['confidence_level'] = np.select(conditions, choices)
    
    return df

# # Usage
# calibrate_fn, optimal_T = fit_temperature_on_dataframe(df_val)
# df_val = apply_calibration_to_dataframe(df_val, calibrate_fn)
# df_test = apply_calibration_to_dataframe(df_test, calibrate_fn)

# print(f"\nTest set confidence distribution:")
# print(df_test['confidence_level'].value_counts(normalize=True))


# -------
# usage
# ------
# model_path = "/PHShome/ea/runs/20250821/144440_650887/checkpoints/model_best_full_model.pt"
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = torch.load(model_path, map_location=device)
# model.eval()

# # 1) Metrics
# metrics, probs, y = evaluate_multiclass_metrics_from_loader(model, val_loader, device=device, use_amp=True)
# print(metrics)

# # 2) Decision curve (class 2 vs {0,1})
# plot_decision_curve_analysis_from_loader(model, val_loader, device=device, use_amp=True, positive_class=2)

# # 3) ROC & PR (all classes on one plot each)
# plot_multiclass_roc_from_loader(model, val_loader, device=device, use_amp=True, class_names=("Never (0)","Low (1)","High (2)"))
# plot_multiclass_pr_from_loader(model, val_loader, device=device, use_amp=True, class_names=("Never (0)","Low (1)","High (2)"))

# # 4) Calibration (pre-TS)
# calib = evaluate_calibration_from_loader(model, val_loader, device=device, use_amp=True, n_bins=10, strategy="quantile", plot=True)
# print({"brier": calib["brier"], "ece_per_class": calib["ece_per_class"]})

# # 5) Optional: Temperature scaling, then re-evaluate calibration
# calibrate_probs, nll_before, nll_after = fit_temperature_on_loader(model, val_loader, device=device, use_amp=True)
# print({"NLL_before": nll_before, "NLL_after": nll_after})
# # Apply calibrated probs for downstream analysis if desired:
# # logits_np, y_np = collect_logits_and_targets(model, val_loader, device=device)
# # probs_cal = calibrate_probs(logits_np)

import torch
import numpy as np

# ---------- helpers ----------
@torch.no_grad()
def _minmax(x, eps=1e-8):
    x = x - x.min()
    d = x.max().clamp_min(eps)
    return x / d

def _attn_to_vec(attn: torch.Tensor | None, device) -> torch.Tensor:
    """
    Canonicalize any attention tensor to a length-12 vector (normalized).
    Accepts None, [B,12,1], [B,12], [12,1], [1,12], [12], or any tensor containing a 12-dim.
    Falls back to uniform if not found or sum≈0.
    """
    if attn is None:
        return torch.full((12,), 1.0/12.0, device=device)
    a = attn.detach().to(device=device, dtype=torch.float32).squeeze()
    sizes = list(a.size())
    if 12 in sizes:
        axis = sizes.index(12)
        v = a.transpose(0, axis).reshape(12, -1).mean(dim=1)
    elif a.numel() == 12:
        v = a.reshape(12)
    else:
        return torch.full((12,), 1.0/12.0, device=device)
    s = v.sum().abs()
    if s < 1e-8:
        return torch.full((12,), 1.0/12.0, device=device)
    return v / s

def _extract_logits_and_attn(model_out):
    """
    Returns (cls_logits, attn_or_None).
    Works with dict (keys: 'cls_logits', 'attn'/'lead_attn'), tuple/list (first=logits; search for a tensor with dim==12),
    or plain tensor (logits only).
    """
    attn = None
    if isinstance(model_out, dict):
        logits = model_out["cls_logits"]
        attn = model_out.get("attn", model_out.get("lead_attn", None))
        return logits, attn
    if isinstance(model_out, (tuple, list)):
        logits = model_out[0]
        for x in model_out[1:]:
            if torch.is_tensor(x) and (12 in x.size()):
                attn = x
                break
        return logits, attn
    if torch.is_tensor(model_out):
        return model_out, None
    raise ValueError("Unrecognized model output for (cls_logits, attn).")

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

@torch.no_grad()
def _minmax(x, eps=1e-8):
    x = x - x.min()
    d = x.max().clamp_min(eps)
    return x / d

def _extract_logits_and_attn(model_out):
    """
    Accepts:
      - dict with keys 'cls_logits' and optionally 'attn'/'lead_attn'
      - tuple/list: assumes first is cls_logits; finds attn tensor by shape [B,12,1] if present
      - tensor: treats as cls_logits only, attn=None
    Returns:
      (cls_logits, attn) where:
        cls_logits: [B, C]
        attn: [B, 12, 1] or None
    """
    attn = None
    if isinstance(model_out, dict):
        logits = model_out["cls_logits"]
        attn = model_out.get("attn", model_out.get("lead_attn", None))
        return logits, attn

    if isinstance(model_out, (tuple, list)):
        logits = model_out[0]
        for x in model_out[1:]:
            if torch.is_tensor(x) and x.ndim == 3 and x.shape[1] == 12 and x.shape[2] == 1:
                attn = x
                break
        return logits, attn

    if torch.is_tensor(model_out):
        return model_out, None

    raise ValueError("Unrecognized model output format for extracting (cls_logits, attn).")

def lead_attention_gradmap_multiclass(
    model,
    ecg,
    target_class: int = 2,      # 0: none, 1: low, 2: high
    device: str = "cuda",
    use_margin: bool = False,   # if True, backprop logit[c] - max_others
):
    """
    Returns:
      saliency: np.ndarray [12, T] lead-wise temporal heatmap in [0,1]
      attn:     np.ndarray [12] attention weights per lead (sum ~ 1) or uniform if missing
    Notes:
      - Works for any C>=3 (infers from logits).
      - Modulates input-gradient saliency by per-lead attention weights (if provided by model).
    """
    model.eval()
    ecg = ecg.to(device).float()
    if ecg.dim() != 3 or ecg.shape[0] != 1:
        raise ValueError("Provide a single ECG as [1,12,T] or [1,T,12].")

    # Ensure channel-first [1,12,T]
    if ecg.shape[1] != 12 and ecg.shape[2] == 12:
        ecg = ecg.permute(0, 2, 1).contiguous()

    # Enable gradient on the exact tensor passed to model
    ecg = ecg.detach().clone().requires_grad_(True)

    # Forward pass: your model returns multiple heads; we grab (cls_logits, attn)
    out = model(ecg)
    logits, attn = _extract_logits_and_attn(out)  # logits [1,C], attn [1,12,1] or None
    C = logits.shape[-1]
    if not (0 <= target_class < C):
        raise ValueError(f"target_class={target_class} not in [0, {C-1}]")

    # Pick target scalar
    if use_margin and C > 1:
        others = torch.cat([logits[:, :target_class], logits[:, target_class+1:]], dim=1)
        margin = logits[:, target_class] - others.max(dim=1, keepdim=False).values
        target_scalar = margin.squeeze()
    else:
        target_scalar = logits[0, target_class]

    # Backprop to input
    model.zero_grad(set_to_none=True)
    if ecg.grad is not None:
        ecg.grad.zero_()
    target_scalar.backward()

    # Input-gradient saliency per lead/time
    grad = ecg.grad  # [1,12,T]
    sal = grad.abs()[0]  # [12,T]

    # Lead attention
    if attn is not None:
        attn_vec = attn[0, :, 0].detach()
    else:
        # fallback to uniform if model doesn't expose lead attention
        attn_vec = torch.full((12,), 1.0 / 12.0, device=sal.device)

    # Modulate per-lead saliency by attention
    sal = sal * attn_vec.view(-1, 1)

    # Normalize each lead to [0,1]
    sal_np = torch.stack([_minmax(sal[i]) for i in range(sal.size(0))], dim=0).cpu().numpy()
    attn_np = attn_vec.detach().cpu().numpy()
    return sal_np, attn_np
