import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)

from utils.data_utils import create_master_df, load_with_csv
from utils.emg_processing_utils import add_baseline_features


def main():
    # raw_data = plot_emg_signals()
    # data = load_with_csv()
    # signal_analysis_pipeline(data)
    # df = create_master_df(data)

    df = pd.read_csv("./data/master_df.csv")

    df["rep_duration"] = df["end"] - df["start"]

    df = df.groupby("file_id", group_keys=False).apply(add_baseline_features)

    y = df["is_fatigued"].astype(int)
    groups = df["file_id"]

    drop_cols = ["is_fatigued", "file_id"]
    X = df.drop(columns=drop_cols)

    # --- model (strong baseline) ---
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear"))
    ])

    # --- group-aware out-of-fold predictions ---
    cv = GroupKFold(n_splits=5)
    oof_proba = cross_val_predict(model, X, y, groups=groups, cv=cv, method="predict_proba")[:, 1]

    # pick a threshold that maximizes balanced accuracy (you can change this objective)
    thresholds = np.linspace(0.05, 0.95, 181)
    baccs = [balanced_accuracy_score(y, (oof_proba >= t).astype(int)) for t in thresholds]
    best_t = thresholds[int(np.argmax(baccs))]

    oof_pred = (oof_proba >= best_t).astype(int)

    print("OOF best threshold:", round(float(best_t), 3))
    print("OOF Balanced Acc:", round(float(balanced_accuracy_score(y, oof_pred)), 3))
    print("OOF ROC AUC:", round(float(roc_auc_score(y, oof_proba)), 3))
    print("OOF PR AUC:", round(float(average_precision_score(y, oof_proba)), 3))
    print("Confusion matrix:\n", confusion_matrix(y, oof_pred))
    print(classification_report(y, oof_pred, digits=3))


if __name__ == "__main__":
    main()
