import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_acc(model, X_flattened, y_labels):
    """
    Calculate accuracy for the model.
    
    Args:
        model: model used to predict 
        X_flattened (np.ndarray): data to predict, shape=(N * D)
        y_labels (np.ndarray): ground truth labels, shape=(N, )

    Returns:
        float: accuracy of prediction (percentage)
        np.ndarray: indices where pred == label (correct predictions)
        np.ndarray: indices where pred != label (mismatches)
    """
    y_pred = model.predict(X_flattened)
    mismatch = y_pred != y_labels
    acc = 1 - np.mean(mismatch)
    return acc * 100, np.where(y_pred == y_labels)[0], np.where(y_pred != y_labels)[0]

def precision_recall(model, X_flattened, y_labels, return_each_class=False):
    """
    Compute the confusion matrix, precision, recall and f1 score.

    Args:
        model: model used to predict
        X_flattened (np.ndarray): data to predict, shape=(N * D)
        y_labels (np.ndarray): ground truth labels, shape=(N, )
        return_each_class (bool, optional): whether to return metrics for each class. Defaults to False.

    Returns:
        conf_matrix: D * D confusion matrix
        precision: (D,) or a single value
        recall: (D,) or a single value
        f1: (D,) or a single value
    """
    avg = None if return_each_class else 'macro'
    y_pred = model.predict(X_flattened)
    conf_matrix = confusion_matrix(y_labels, y_pred)
    precision = precision_score(y_labels, y_pred, average=avg, zero_division=np.nan)
    recall = recall_score(y_labels, y_pred, average=avg, zero_division=np.nan)
    f1 = f1_score(y_labels, y_pred, average=avg, zero_division=np.nan)
    return conf_matrix, precision, recall, f1
    