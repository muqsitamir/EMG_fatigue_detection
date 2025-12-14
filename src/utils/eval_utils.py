from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix, classification_report, \
    average_precision_score

from src.utils.ml_utils import onset_from_labels, trigger_from_proba, trigger_from_proba_m_of_n


def evaluate_onset_timing(df, proba, thr=0.58, M=2, N=3, m_of_n=False,
                          group_col="file_id", order_col="rep",
                          label_col="is_fatigued"):
    d = df.copy()
    d["proba"] = proba

    # ensure proper ordering within each file
    if order_col in d.columns:
        d = d.sort_values([group_col, order_col]).reset_index(drop=True)
    else:
        # fallback: sort by start if you have it
        if "start" in d.columns:
            d = d.sort_values([group_col, "start"]).reset_index(drop=True)
        else:
            d = d.sort_values([group_col]).reset_index(drop=True)

    rows = []
    for fid, g in d.groupby(group_col):
        g = g.reset_index(drop=True)

        onset_idx = onset_from_labels(g, label_col)
        if m_of_n:
            trig_idx = trigger_from_proba_m_of_n(g, "proba", thr=thr, M=M, N=N)
        else:
            trig_idx  = trigger_from_proba(g, "proba", thr=thr, M=M)

        onset_rep = int(g.loc[onset_idx, order_col]) if (onset_idx is not None and order_col in g.columns) else onset_idx
        trig_rep  = int(g.loc[trig_idx,  order_col]) if (trig_idx  is not None and order_col in g.columns) else trig_idx

        delta = None
        if onset_idx is not None and trig_idx is not None:
            delta = trig_rep - onset_rep

        rows.append({
            "file_id": fid,
            "n_reps": len(g),
            "onset_rep": onset_rep,
            "trigger_rep": trig_rep,
            "delta_reps (trigger - onset)": delta,
            "triggered": trig_rep is not None,
        })

    out = pd.DataFrame(rows)

    valid = out.dropna(subset=["delta_reps (trigger - onset)"])
    summary = {}
    if len(valid):
        delta = valid["delta_reps (trigger - onset)"].to_numpy()
        summary = {
            "files_total": len(out),
            "files_with_onset_and_trigger": len(valid),
            "never_triggered": int((out["triggered"] == False).sum()),
            "mean_delta_reps": float(np.mean(delta)),
            "median_delta_reps": float(np.median(delta)),
            "mae_reps": float(np.mean(np.abs(delta))),
            "pct_within_0_reps": float(np.mean(delta == 0)),
            "pct_within_1_rep": float(np.mean(np.abs(delta) <= 1)),
            "pct_within_2_reps": float(np.mean(np.abs(delta) <= 2)),
            "early_rate": float(np.mean(delta < 0)),
            "late_rate": float(np.mean(delta > 0)),
        }

    return out, summary


def evaluate_predictions(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float
) -> Dict:
    y_pred = (proba >= threshold).astype(int)

    metrics = {
        "threshold": threshold,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, digits=3),
        "y_pred": y_pred,
    }
    return metrics

