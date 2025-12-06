import pandas as pd

from utils.eval_utils import evaluate_predictions
from utils.ml_utils import TrainConfig, make_xy_groups, build_model, train_oof_predict_proba, select_threshold_max_bacc
from typing import Dict

from utils.plot_utils import plot_threshold_sweep, plot_confusion, plot_proba_hist, plot_roc_pr


def run_training_eval(
    df: pd.DataFrame,
    cfg: TrainConfig,
    label_col: str = "is_fatigued",
    group_col: str = "file_id",
    plot: bool = True
) -> Dict:
    X, y, groups = make_xy_groups(df, label_col=label_col, group_col=group_col)
    model = build_model()

    oof_proba = train_oof_predict_proba(X, y, groups, model, cfg)
    best_t, sweep = select_threshold_max_bacc(y, oof_proba, cfg.threshold_grid)
    results = evaluate_predictions(y, oof_proba, best_t)
    results["threshold_sweep"] = sweep
    results["oof_proba"] = oof_proba

    if plot:
        plot_threshold_sweep(sweep, best_t)
        plot_roc_pr(y, oof_proba)
        plot_confusion(results["confusion_matrix"])
        plot_proba_hist(y, oof_proba)

    return results, best_t
