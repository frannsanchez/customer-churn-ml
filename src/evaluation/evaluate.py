from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

POS_LABEL = "Yes"


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=["No", POS_LABEL]).ravel()
    return {
        "precision": precision_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=POS_LABEL),
        "f1": f1_score(y_true, y_pred, pos_label=POS_LABEL),
        "roc_auc": roc_auc_score((y_true == POS_LABEL).astype(int), y_prob),
        # Matriz de confusión desglosada: fn = clientes que se van y no detectamos
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def print_metrics(name: str, metrics: dict) -> None:
    print(f"\n--- {name} ---")
    for key, value in metrics.items():
        formatted = value if isinstance(value, int) else f"{value:.4f}"
        print(f"{key.capitalize()}: {formatted}")


def print_confusion_matrix(y_true, y_pred) -> None:
    print("Matriz de confusión (filas: real No/Yes, columnas: predicho No/Yes):")
    print(confusion_matrix(y_true, y_pred, labels=["No", POS_LABEL]))