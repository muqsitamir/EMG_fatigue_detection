import pandas as pd

from emg_fatigue_detection.utils.data_utils import save_model_bundle
from emg_fatigue_detection.utils.eval_utils import evaluate_predictions
from emg_fatigue_detection.utils.ml_utils import TrainConfig, make_xy_groups, build_model, train_oof_predict_proba, select_threshold_max_bacc
from typing import Dict

from emg_fatigue_detection.utils.plot_utils import plot_threshold_sweep, plot_confusion, plot_proba_hist, plot_roc_pr


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


def train_final_model(
    df: pd.DataFrame,
    best_threshold: float,m,n
    ):
    y = df["is_fatigued"].astype(int).to_numpy()
    X = df.drop(columns=["is_fatigued", "file_id"]).select_dtypes(include=["number"])
    feature_cols = list(X.columns)

    final_model = build_model()
    final_model.fit(X, y)

    best_threshold = best_threshold
    trigger_m, trigger_n = m, n
    smooth_alpha = None

    save_model_bundle(
        model=final_model,
        feature_cols=feature_cols,
        best_threshold=best_threshold,
        trigger_M=trigger_m,
        trigger_N=trigger_n,
        smooth_alpha=smooth_alpha,
        bundle_path="./models/fatigue_model_bundle.joblib"
    )