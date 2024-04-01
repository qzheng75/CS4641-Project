import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold


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


def visualize_confusion_matrices(confusion_matrices, class_names=None, titles=None, grand_title='Confusion matrices', cmap='Blues'):
    if class_names is None:
        class_names = ['blues', 'classical', 'country', 'disco', 'hiphop',
                       'jazz', 'metal', 'pop', 'reggae', 'rock']

    # Set titles to split names if not provided
    if titles is None:
        match len(confusion_matrices):
            case 1: titles = ('Full dataset')
            case 2: titles = ('Train', 'Test')
            case 3: titles = ('Train', 'Val', 'Test')

    num_subplots = len(confusion_matrices)
    fig, axes = plt.subplots(1, num_subplots, figsize=(6 * num_subplots, 6))

    for idx in range(len(confusion_matrices)):
        confusion_mat = confusion_matrices[idx]
        ax = axes[idx] if num_subplots > 1 else axes

        im = ax.imshow(confusion_mat, interpolation='nearest', cmap=cmap)

        # Add a color bar
        cbar = ax.figure.colorbar(im, ax=ax, shrink=0.7)

        # Set the tick marks and labels
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)

        # Add labels to each cell
        thresh = confusion_mat.max() / 2.
        for i in range(confusion_mat.shape[0]):
            for j in range(confusion_mat.shape[1]):
                ax.text(j, i, format(confusion_mat[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if confusion_mat[i, j] > thresh else "black")

        # Set the axis labels and title
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(titles[idx])

    plt.suptitle(grand_title)
    fig.tight_layout()

def kfold_validation(model, data, label, n_splits, random_state = 42, shuffle=True):
    accs = []
    precisions = []
    recalls = []
    f1s = []
    kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    for train_index, test_index in kf.split(data, label):
        X_train, X_test, y_train, y_test = data[train_index], data[test_index], label[train_index], label[test_index]
        model.fit(X_train, y_train)
        acc,_,_ = calculate_acc(model, X_test, y_test)
        _, precision, recall, f1 = precision_recall(model, X_test, y_test, return_each_class=False)
        accs.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    return np.mean(accs), np.mean(precisions), np.mean(recalls), np.mean(f1s)
