from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics
    """

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "f1_Score": f1_score(y_true, y_pred),
    }
    cr = classification_report(y_true, y_pred, target_names=["Healthy", "Unhealthy"])
    cm = confusion_matrix(y_true, y_pred)

    return metrics, cm, cr
