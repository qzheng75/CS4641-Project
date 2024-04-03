import sys
import os

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score

import torch
from torch import nn
from torch.utils.data import TensorDataset
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms

from audio_toolbox.models import OneHotCrossEntropyLoss
from audio_toolbox.metrics import audio_dataset_split, precision_recall
from audio_toolbox.trainer import ModelTrainer


def sparse_labels_to_tensor(labels):
    scipy_sparse_matrix = csr_matrix(labels)

    # Convert the scipy sparse matrix to a torch sparse tensor
    indices = torch.LongTensor([scipy_sparse_matrix.indices, scipy_sparse_matrix.indptr[:-1]])
    values = torch.from_numpy(scipy_sparse_matrix.data)
    shape = scipy_sparse_matrix.shape

    return torch.sparse_coo_tensor(indices, values, shape).to_dense()


if __name__ == '__main__':
    data = torch.load('./processed_data/complete_spectrogram_dataset/processed_data.pt')
    label = torch.load('./processed_data/complete_spectrogram_dataset/processed_label.pt')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    RANDOM_STATE = 42
    X_train, y_train, X_val, y_val,\
    X_test, y_test = audio_dataset_split(data, label, train_val_test_ratio=(0.09, 0.005, 0.005), random_state=RANDOM_STATE)
    
    X_train = torch.tensor(X_train, dtype=torch.float).permute(0, 3, 1, 2).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float).permute(0, 3, 1, 2).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float).permute(0, 3, 1, 2).to(device)
    y_train = sparse_labels_to_tensor(y_train).to(device)
    y_val = sparse_labels_to_tensor(y_val).to(device)
    y_test = sparse_labels_to_tensor(y_test).to(device)

    # Define the preprocessing transforms
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    for t in preprocess.transforms:
        X_train = preprocess(X_train)
        X_val = preprocess(X_val)
        X_test = preprocess(X_test)
    
    
    model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg16', pretrained=True)
    model.features[0] = nn.Conv2d(4, 64, kernel_size=3, padding=1)
    num_classes = 10
    model.classifier[6] = nn.Linear(4096, num_classes)
    model.to(device)

    datasets = {
        'train': TensorDataset(X_train, y_train),
        'val': TensorDataset(X_val, y_val),
        'test': TensorDataset(X_test, y_test)
    }

    loss_fn = OneHotCrossEntropyLoss()
    learning_rate = 1e-3  # Adjust the learning rate as needed
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    trainer = ModelTrainer(datasets, model, loss_fn, optimizer, scheduler)
    trainer_config = {
        'save': False,
        'num_epochs': 100,
        'batch_size': 32,
        'batch_tqdm': True
    }

    trainer.train(**trainer_config)
    train_res = trainer.predict(X_train)
    val_res = trainer.predict(X_val)
    test_res = trainer.predict(X_test)

    print(f"Train accuracy: {100 * accuracy_score(train_res, y_train):.2f}%")
    print(f"Validation accuracy: {100 * accuracy_score(val_res, y_val):.2f}%")
    print(f"Test accuracy: {100 * accuracy_score(test_res, y_test):.2f}%")

    _, _, _, f1_train = precision_recall(trainer, X_train, y_train)
    _, _, _, f1_val = precision_recall(trainer, X_val, y_val)
    _, _, _, f1_test = precision_recall(trainer, X_test, y_test)
        
    print(f"Train f1 score: {f1_train:.4f}")
    print(f"Validation f1 score: {f1_val:.4f}")
    print(f"Test f1 score: {f1_test:.4f}")
    
    