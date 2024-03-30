import numpy as np
import torch

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score


def audio_dataset_split(data, label, train_val_test_ratio=None, random_state=None):
    if train_val_test_ratio is None:
        train_val_test_ratio = (0.9, 0.05, 0.05)
    
    if random_state is not None:
        torch.manual_seed(random_state)
    
    n = data.shape[0]
    indices = torch.randperm(n)
    
    train_size = int(train_val_test_ratio[0] * n)
    val_size = int(train_val_test_ratio[1] * n)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    train_data, val_data, test_data = data[train_indices], data[val_indices], data[test_indices]
    train_label, val_label, test_label = label[train_indices], label[val_indices], label[test_indices]
    
    return train_data, train_label, val_data, val_label, test_data, test_label
    

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
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_labels, torch.Tensor):
        y_labels = y_labels.cpu().numpy()
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
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_labels, torch.Tensor):
        y_labels = y_labels.cpu().numpy()
    conf_matrix = confusion_matrix(y_labels, y_pred)
    precision = precision_score(y_labels, y_pred, average=avg, zero_division=np.nan)
    recall = recall_score(y_labels, y_pred, average=avg, zero_division=np.nan)
    f1 = f1_score(y_labels, y_pred, average=avg, zero_division=np.nan)
    return conf_matrix, precision, recall, f1
    