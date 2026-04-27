from scipy.special import softmax
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
from ecg2dig.utils.datasets import is_valid_ecg, is_bound_ecg, safe_pearsonr
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, average_precision_score
from tqdm import tqdm
from typing import Any, Callable, Dict, Optional, Tuple
from ecg2dig.utils.helpers import (parse_args_from_training_log,
                                   plot_roc_curves,
                                   compute_f1, 
                                   metrics_at_threshold,
                                   threshold_for_target_specificity, 
                                   subgroup_analysis, 
                                   plot_roc_curves)

def _run_forward_pass(model, loader, device, label_col):
    """Run model over loader, return list of per-sample result dicts."""
    model.eval()
    results = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            if batch is None:
                continue
            (xb,         # ecg tensor [B, T, C] or [B, C, T]
             dig_tensor, # digoxin numeric value
             y_cls,      # target value - class digoxin 
             female_tensor, 
             age_tensor,
             MRN_path,   # # tuple of strings (length B)
             row_index) = batch # tensor/tuple of ints (length B)
            
            xb = xb.to(device)

            cls_logits, _attn, hr_out, dig_out, _, _ = model(xb)
            cls_logits = cls_logits.detach().cpu()
            hr_out = hr_out.squeeze(-1).cpu()
            dig_out = dig_out.squeeze(-1).cpu()

            for i in range(xb.size(0)):
                results.append({
                    "row_index":  int(row_index[i]),
                    "MRN_path":   MRN_path[i],
                    "cls_logits": cls_logits[i].tolist(),
                    "hr_pred":    float(hr_out[i]),
                    "dig_pred":   float(dig_out[i]),
                    "dig_true":   float(dig_tensor[i].item()),
                    label_col:    int(y_cls[i].item()),
                })
    return results


def _merge_predictions(loader, results):
    """Merge predictions back onto the original dataframe via row_index."""
    pred_df = pd.DataFrame(results)
    orig_df = loader.dataset.df.copy()
    orig_df['row_index'] = orig_df.index
    full_df = (orig_df.merge(pred_df, on="row_index", how="inner",
                             suffixes=('', '_pred'))
                      .sort_values("row_index")
                      .reset_index(drop=True))
    return full_df


def _compute_classification_metrics(y_true, probs):
    """
    Returns (macro_auc, macro_f1, per_class_auc, per_class_f1, per_class_ap,
             f1_reasons, macro_auc_note).
    Macro AUC is averaged over per-class OVR AUCs for classes actually
    present with both positives and negatives.
    """
    y_true = np.asarray(y_true).astype(int)
    C = probs.shape[1]

    per_class_auc, per_class_ap = {}, {}
    auc_skip_notes = []

    # Per-class OVR AUC and AP — only for classes with both pos and neg samples.
    for c in range(C):
        y_bin = (y_true == c).astype(int)
        n_pos, n_neg = int(y_bin.sum()), int(len(y_bin) - y_bin.sum())
        if n_pos == 0:
            auc_skip_notes.append(f"class {c}: no positive samples (skipped)")
            continue
        if n_neg == 0:
            auc_skip_notes.append(f"class {c}: no negative samples (skipped)")
            continue
        try:
            per_class_auc[c] = float(roc_auc_score(y_bin, probs[:, c]))
            per_class_ap[c] = float(average_precision_score(y_bin, probs[:, c]))
        except Exception as e:
            auc_skip_notes.append(f"class {c}: AUC/AP failed ({e})")

    # Macro AUC = mean over present per-class AUCs (option 2).
    if per_class_auc:
        macro_auc = float(np.mean(list(per_class_auc.values())))
        macro_auc_note = (f"averaged over {len(per_class_auc)}/{C} present classes"
                          if len(per_class_auc) < C else "")
    else:
        macro_auc = float('nan')
        macro_auc_note = "no classes had both pos and neg samples"

    # F1
    y_pred = np.argmax(probs, axis=1)
    per_class_f1, macro_f1, f1_reasons = compute_f1(y_true, y_pred, C)

    return (macro_auc, macro_f1, per_class_auc, per_class_f1, per_class_ap,
            f1_reasons, macro_auc_note, auc_skip_notes)


def _print_metrics(macro_auc, macro_f1, per_class_auc, per_class_f1,
                   per_class_ap, f1_reasons, macro_auc_note, auc_skip_notes,
                   dig_pearson_r):
    print(f"Digoxin ROC AUC (macro): {macro_auc:.4f}"
          + (f"  [{macro_auc_note}]" if macro_auc_note else ""))
    print(f"Digoxin Macro F1:        {macro_f1:.4f}")

    if per_class_auc:
        pcs = " | ".join(f"class{c}: {a:.4f}" for c, a in sorted(per_class_auc.items()))
        print(f"Per-class AUC (OVR):     {pcs}")
    if per_class_f1:
        pcsf = " | ".join(f"class{c}: {f:.4f}" for c, f in sorted(per_class_f1.items()))
        print(f"Per-class F1:            {pcsf}")
    if per_class_ap:
        pcsap = " | ".join(f"class{c}: {ap:.4f}" for c, ap in sorted(per_class_ap.items()))
        print(f"Per-class AP:            {pcsap}")

    print(f"DIG Pearson r:           {dig_pearson_r:.4f}")

    notes = auc_skip_notes + f1_reasons
    if notes:
        print("Notes:")
        for n in notes:
            print(f"  - {n}")


def inference(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    use_multi_class: bool,
    loader_name: str = "val_loader",
    output_dir: Optional[str] = None,
):
    
    """
    Run inference over `loader`, compute metrics, optionally save a results CSV.

    Dataset is assumed to yield 5-tuples per sample (meta_columns=[], no intervals):
        (ecg_tensor, dig_level_tensor, high_dig_tensor, MRN_path, row_index)

    Label column convention:
      - use_multi_class=True  -> 3-class labels stored in column 'dig_class_true'  (0/1/2)
      - use_multi_class=False -> binary labels stored in column 'high_dig_true' (0/1)

    Prints:
      - Digoxin ROC AUC (macro, OVR)
      - Digoxin Macro F1
      - Per-class OVR AUC, F1, AP
      - DIG Pearson r (regression head)

    Returns:
      full_df: merged DataFrame of loader.dataset.df + per-row predictions
    """

    model.eval()
    results = []
    
    print(f'--------\nINFERENCE\n-----------\n')
    # choose which column name to write the integer class label into
    label_col = "dig_class_true" if use_multi_class else "high_dig_true" 
    
    print(f"[DEBUG] use_multi_class = {use_multi_class!r}")
    print(f"[DEBUG] label_col = {label_col!r}")

    # 1) Forward pass
    results = _run_forward_pass(model, loader, device, label_col)
    if not results:
        raise RuntimeError(
            "No samples processed — every batch was dropped. "
            "Check ECGDrugDataset.__getitem__ for load failures."
        )
        
    # Merge with original dataframe to keep any contextual columns
    # 2) Merge back with the loader's dataframe
    full_df = _merge_predictions(loader, results)
    print(f"full_df created: {full_df.shape}")
    
    # 3) Classification metrics
    cls_logits_arr = np.vstack(full_df["cls_logits"].values)
    probs = softmax(cls_logits_arr, axis=1)
    y_true = full_df[label_col].to_numpy()
    (macro_auc, macro_f1, per_class_auc, per_class_f1, per_class_ap,
     f1_reasons, macro_auc_note, auc_skip_notes) = _compute_classification_metrics(y_true, probs)
    
    # 4) Regression metric
    try:
        r, _ = safe_pearsonr(full_df["dig_true"].to_numpy(),
                             full_df["dig_pred"].to_numpy())
        dig_pearson_r = float(r)
    except Exception as e:
        print(f"DIG Pearson r: failed ({e})")
        dig_pearson_r = float('nan')

    # 5) Report
    _print_metrics(macro_auc, macro_f1, per_class_auc, per_class_f1,
                   per_class_ap, f1_reasons, macro_auc_note, auc_skip_notes,
                   dig_pearson_r)

    # 6) Save
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"infer_{loader_name}_df.csv")
        full_df.to_csv(out_path, index=False)
        print(f"Saved inference table to: {out_path}")

    return full_df

