import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve


def plot_threshold_sweep(sweep: pd.DataFrame, best_t: float) -> None:
    plt.figure()
    plt.plot(sweep["threshold"], sweep["balanced_accuracy"])
    plt.axvline(best_t)
    plt.xlabel("Threshold")
    plt.ylabel("Balanced Accuracy")
    plt.title("Threshold sweep (maximize balanced accuracy)")
    plt.show()


def plot_roc_pr(y_true: np.ndarray, proba: np.ndarray) -> None:
    fpr, tpr, _ = roc_curve(y_true, proba)
    prec, rec, _ = precision_recall_curve(y_true, proba)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.show()

    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve")
    plt.show()


def plot_confusion(cm: np.ndarray) -> None:
    plt.figure()
    plt.imshow(cm)
    plt.xticks([0, 1], ["Not fatigued", "Fatigued"])
    plt.yticks([0, 1], ["Not fatigued", "Fatigued"])
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # annotate
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")

    plt.title("Confusion Matrix")
    plt.show()


def plot_proba_hist(y_true: np.ndarray, proba: np.ndarray) -> None:
    plt.figure()
    plt.hist(proba[y_true == 0], bins=30, alpha=0.7, label="Not fatigued")
    plt.hist(proba[y_true == 1], bins=30, alpha=0.7, label="Fatigued")
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title("OOF predicted probabilities")
    plt.legend()
    plt.show()