from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

POS_LABEL = "Yes"


def compute_metrics(y_true, y_pred, y_prob) -> dict:
        return {
        "precision": precision_score(y_true, y_pred, pos_label=POS_LABEL),
        "recall": recall_score(y_true, y_pred, pos_label=POS_LABEL),
        "f1": f1_score(y_true, y_pred, pos_label=POS_LABEL),
        "roc_auc": roc_auc_score((y_true == POS_LABEL).astype(int), y_prob),
    }


def print_metrics(name: str, metrics: dict) -> None:
    print(f"\n--- {name} ---")
    for key, value in metrics.items():
        print(f"{key.capitalize()}: {value:.4f}")


def print_confusion_matrix(y_true, y_pred) -> None:
    print("Matriz de confusión:")
    print(confusion_matrix(y_true, y_pred))