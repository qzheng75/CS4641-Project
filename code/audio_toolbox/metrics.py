import numpy as np
import os
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from collections import Counter


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
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

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
    precision = precision_score(y_labels, y_pred, average=avg)
    recall = recall_score(y_labels, y_pred, average=avg)
    f1 = f1_score(y_labels, y_pred, average=avg)
    return conf_matrix, precision, recall, f1


def kfold_validation(model, data, label, n_splits, random_state=42, shuffle=True):
    accs = []
    precisions = []
    recalls = []
    f1s = []
    kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    for train_index, test_index in kf.split(data, label):
        X_train, X_test, y_train, y_test = data[train_index], data[test_index], label[train_index], label[test_index]
        model.fit(X_train, y_train)
        acc, _, _ = calculate_acc(model, X_test, y_test)
        _, precision, recall, f1 = precision_recall(model, data, label, return_each_class=False)
        accs.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return np.mean(accs), np.mean(precisions), np.mean(recalls), np.mean(f1s)


def splitPredicting(model, data, dataSet):
    """
    first find the sliced 10 data for each data in the testing data, then do the prediction to all 10 data.
    Pick the mode of the prediction to be the final prediction, then compute the accuracy of this prediction.
    Note that the first slice of filename "blues.00000.wav" is named as "blues.00000.0.wav"
    Args:
        model: the model we trained
        data: testing data
        dataSet: The whole data set include the testing data
    Returns:
        result: the predicted label
    """
    filenames = data['filename']
    prediction = []
    for i in range(len(filenames)):
        slicedSong = []
        for j in range(10):
            song = filenames.iloc[i]
            filename, extension = os.path.splitext(song)
            slices = f"{filename}.{j}{extension}"
            sliceRow = dataSet[dataSet['filename'] == slices].drop(['filename', 'length', 'label'], axis=1)
            if sliceRow.shape[0] != 0:
                slice_prediction = model.predict(sliceRow)[0]
                slicedSong.append(slice_prediction)
        prediction.append(Counter(slicedSong).most_common(1)[0][0])
    return np.array(prediction)


def splitTabularPredicting(model, data, dataSet):
    """
    first find the sliced 10 data for each data in the testing data, then do the prediction to all 10 data.
    Pick the mode of the prediction to be the final prediction, then compute the accuracy of this prediction.
    Note that the first slice of filename "blues.00000.wav" is named as "blues.00000.0.wav"
    Args:
        model: the model we trained
        data: testing data
        dataSet: The whole data set include the testing data
    Returns:
        result: the predicted label
    """
    index = data['index']
    prediction = []
    for i in range(len(index)):
        songIndex = index.iloc[i]
        sliceRow = dataSet[dataSet['index'] == songIndex].drop(['label', 'index'], axis=1)
        slice_prediction = model.predict(sliceRow)
        prediction.append(Counter(slice_prediction).most_common(1)[0][0])
    return np.array(prediction)
