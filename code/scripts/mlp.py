import sys
import os

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import torch
from torch import nn
from torch.utils.data import TensorDataset
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from audio_toolbox.metrics import audio_dataset_split, precision_recall
from audio_toolbox.models import SimpleLinearModel
from audio_toolbox.trainer import ModelTrainer


if __name__ == '__main__':
    data = torch.load('processed_data/complete_dataset/processed_data.pt')
    label = torch.load('processed_data/complete_dataset/processed_label.pt')
    
    RANDOM_STATE = 42
    X_train, y_train, X_val, y_val, X_test, y_test = audio_dataset_split(data, label,
                                         train_val_test_ratio=(0.9, 0.05, 0.05), random_state=RANDOM_STATE)
    
    X_train_flat = X_train.view(X_train.shape[0], -1)
    X_val_flat = X_val.view(X_val.shape[0], -1)
    X_test_flat = X_test.view(X_test.shape[0], -1)
    
    scaler = StandardScaler()
    pca = PCA(n_components=0.95)
    train_scaled = scaler.fit_transform(X_train_flat)
    val_scaled = scaler.transform(X_val_flat)
    test_scaled = scaler.transform(X_test_flat)

    train_data = torch.tensor(pca.fit_transform(train_scaled), dtype=torch.float)
    val_data = torch.tensor(pca.transform(val_scaled), dtype=torch.float)
    test_data = torch.tensor(pca.transform(test_scaled), dtype=torch.float)

    datasets = {
        'train': TensorDataset(train_data, y_train),
        'val': TensorDataset(val_data, y_val),
        'test': TensorDataset(test_data, y_test)
    }
    
    input_size = train_data.size(1)
    output_size = 10

    model = SimpleLinearModel(input_size, output_size)
    
    loss_fn = nn.CrossEntropyLoss()
    batch_size = 32
    learning_rate = 1e-3  # Adjust the learning rate as needed
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    trainer = ModelTrainer(datasets, model, loss_fn, optimizer, scheduler)
    trainer_config = {
        'save': False,
        'num_epochs': 100,
        'batch_size': batch_size
    }
    trainer.train(**trainer_config)
    
    train_res = trainer.predict(train_data)
    val_res = trainer.predict(val_data)
    test_res = trainer.predict(test_data)
    print(f"Train accuracy: {100 * accuracy_score(train_res, y_train):.2f}%")
    print(f"Validation accuracy: {100 * accuracy_score(val_res, y_val):.2f}%")
    print(f"Test accuracy: {100 * accuracy_score(test_res, y_test):.2f}%")

    _, _, _, f1_train = precision_recall(trainer, train_data, y_train)
    _, _, _, f1_val = precision_recall(trainer, val_data, y_val)
    _, _, _, f1_test = precision_recall(trainer, test_data, y_test)
    
    print(f"Train f1 score: {f1_train:.4f}")
    print(f"Validation f1 score: {f1_val:.4f}")
    print(f"Test f1 score: {f1_test:.4f}")