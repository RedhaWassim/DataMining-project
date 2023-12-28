# general metrics
import numpy as np


def accuracy(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    return correct / len(y_true)


def precision(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positives = np.sum(y_pred == 1)
    return true_positives / predicted_positives if predicted_positives != 0 else 0


def recall(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    actual_positives = np.sum(y_true == 1)
    return true_positives / actual_positives if actual_positives != 0 else 0


def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) != 0 else 0


def specificite(y_true, y_pred):
    true_negatives = np.sum((y_true == 0) & (y_pred == 0))
    actual_negatives = np.sum(y_true == 0)
    return true_negatives / actual_negatives if actual_negatives != 0 else 0


def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return [[tn, fp], [fn, tp]]


# metrics per class
def calculate_metrics_per_class(y_true, y_pred, classes):
    metrics = {}
    for cls in classes:
        cls_true = [1 if c == cls else 0 for c in y_true]
        cls_pred = [1 if c == cls else 0 for c in y_pred]

        # Precision (Précision)
        true_positives = sum(
            [1 for true, pred in zip(cls_true, cls_pred) if true == pred == 1]
        )
        predicted_positives = cls_pred.count(1)
        precision = (
            true_positives / predicted_positives if predicted_positives != 0 else 0
        )

        # Recall (Rappel)
        actual_positives = cls_true.count(1)
        recall = true_positives / actual_positives if actual_positives != 0 else 0

        # F1-Score (F-Score)
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) != 0
            else 0
        )

        metrics[cls] = {"precision": precision, "recall": recall, "f1_score": f1}

    return metrics


def specificite_per_class(y_true, y_pred, classes):
    specificity_scores = {}
    for cls in classes:
        true_negatives = sum(
            [1 for true, pred in zip(y_true, y_pred) if true != cls and pred != cls]
        )
        actual_negatives = len([1 for c in y_true if c != cls])
        specificity = true_negatives / actual_negatives if actual_negatives != 0 else 0
        specificity_scores[cls] = specificity
    return specificity_scores
