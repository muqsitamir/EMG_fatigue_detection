import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Tuple, List

from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score
)


@dataclass
class TrainConfig:
    n_splits: int = 5
    threshold_grid: Tuple[float, float, int] = (0.05, 0.95, 181)  # start, end, count
    random_state: int = 42  # used only if you later switch to shuffled splits


def make_xy_groups(
    df: pd.DataFrame,
    label_col: str = "is_fatigued",
    group_col: str = "file_id",
    drop_cols: List[str] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if drop_cols is None:
        drop_cols = [label_col, group_col]

    y = df[label_col].astype(int).to_numpy()
    groups = df[group_col].to_numpy()
    X = df.drop(columns=drop_cols)

    X = X.select_dtypes(include=[np.number])

    return X, y, groups

def build_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="liblinear"
        ))
    ])

def train_oof_predict_proba(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    model: Pipeline,
    cfg: TrainConfig
) -> np.ndarray:
    cv = GroupKFold(n_splits=cfg.n_splits)
    oof_proba = cross_val_predict(
        model, X, y,
        groups=groups,
        cv=cv,
        method="predict_proba"
    )[:, 1]
    return oof_proba

def select_threshold_max_bacc(
    y_true: np.ndarray,
    proba: np.ndarray,
    grid: Tuple[float, float, int]
) -> Tuple[float, pd.DataFrame]:
    t0, t1, n = grid
    thresholds = np.linspace(t0, t1, n)
    baccs = [balanced_accuracy_score(y_true, (proba >= t).astype(int)) for t in thresholds]
    best_idx = int(np.argmax(baccs))
    best_t = float(thresholds[best_idx])

    sweep = pd.DataFrame({
        "threshold": thresholds,
        "balanced_accuracy": baccs
    })
    return best_t, sweep

def onset_from_labels(g, label_col="is_fatigued"):
    idx = np.flatnonzero(g[label_col].to_numpy() == 1)
    return int(idx[0]) if len(idx) else None

def trigger_from_proba(g, proba_col="proba", thr=0.58, M=2):
    above = (g[proba_col].to_numpy() >= thr).astype(int)
    run = 0
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= M:
            return int(i)
    return None


def trigger_from_proba_m_of_n(g, proba_col="proba", thr=0.58, M=2, N=3):
    above = (g[proba_col].to_numpy() >= thr).astype(int)
    for i in range(len(above)):
        win = above[max(0, i - N + 1): i + 1]  # last N (or fewer at start)
        if win.sum() >= M:
            return int(i)  # trigger at i (end of the window)
    return None
