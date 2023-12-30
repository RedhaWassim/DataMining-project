import numpy as np


def multiclass_confusion_matrix(y_true, y_pred, class_label):
    tp = np.sum((y_true == class_label) & (y_pred == class_label))
    fp = np.sum((y_true != class_label) & (y_pred == class_label))
    tn = np.sum((y_true != class_label) & (y_pred != class_label))
    fn = np.sum((y_true == class_label) & (y_pred != class_label))

    return tp, fp, tn, fn


def accuracy(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    return correct / len(y_true)


def precision_recall_f1(y_true, y_pred, class_label):
    tp, fp, _, fn = multiclass_confusion_matrix(y_true, y_pred, class_label)
    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if precision + recall != 0
        else 0
    )
    return precision, recall, f1


def specificity(y_true, y_pred, class_label):
    _, _, tn, fp = multiclass_confusion_matrix(y_true, y_pred, class_label)
    return tn / (tn + fp) if tn + fp != 0 else 0


def micro_average(y_true, y_pred, classes):
    total_tp, total_fp, total_fn = 0, 0, 0
    for class_label in classes:
        tp, fp, _, fn = multiclass_confusion_matrix(y_true, y_pred, class_label)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp != 0 else 0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn != 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if precision + recall != 0
        else 0
    )
    return precision, recall, f1


def macro_average(y_true, y_pred, classes):
    sum_precision, sum_recall, sum_f1 = 0, 0, 0
    for class_label in classes:
        precision, recall, f1 = precision_recall_f1(y_true, y_pred, class_label)
        sum_precision += precision
        sum_recall += recall
        sum_f1 += f1

    n = len(classes)
    return sum_precision / n, sum_recall / n, sum_f1 / n
