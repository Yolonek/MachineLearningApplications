import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from dataclasses import dataclass
from matplotlib import pyplot as plt
from CommonFunctions import enhance_plot


@dataclass
class LearningParameters:
    batch_size: int = 128
    cpu_count: int = os.cpu_count()
    learning_rate: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 0.001
    epochs: int = 150
    device: torch.device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    dropout: float = 0.3
    batch_norm: bool = True


def load_CIFAR10(path, transform, batch_size=32, subset=None, download=False):
    train_data = datasets.CIFAR10(root=path,
                                  train=True,
                                  download=download,
                                  transform=transform)
    test_data = datasets.CIFAR10(root=path,
                                 train=False,
                                 download=download,
                                 transform=transform)
    if subset is not None:
        train_data = Subset(train_data, range(subset[0]))
        test_data = Subset(test_data, range(subset[1]))
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=True)
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 pin_memory=True)
    return train_data, test_data, train_dataloader, test_dataloader


def accuracy(y_pred, y_true):
    return (y_true == y_pred).sum().item() / len(y_pred)


def train_step(model, dataloader, criterion, accuracy_function, optimizer, device):
    train_loss, accuracy = 0, 0
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = criterion(y_pred, y)
        train_loss += loss.item()
        accuracy += accuracy_function(y_pred.argmax(dim=1), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= (data_len := len(dataloader))
    accuracy /= data_len
    return train_loss, accuracy


def test_step(model, dataloader, criterion, accuracy_function, device):
    test_loss, accuracy = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            test_pred = model(X)
            loss = criterion(test_pred, y)
            test_loss += loss.item()
            accuracy += accuracy_function(test_pred.argmax(dim=1), y)
        test_loss /= (data_len := len(dataloader))
        accuracy /= data_len
    return test_loss, accuracy


def plot_loss_and_accuracy(title, epochs, loss_dict, accuracy_dict, file):
    with plt.style.context('cyberpunk'):
        figure, axes = plt.subplots(2, 1, layout='constrained', figsize=(8, 6))
        x_axis = range(1, epochs + 1)
        for loss_type, loss in loss_dict.items():
            axes[0].plot(x_axis, loss, label=f'{loss_type}: {min(loss):.3f}')
        axes[0].set(ylabel='Loss', title=title)
        axes[0].legend()
        enhance_plot(figure, axes[0], glow=True)
        for accuracy_type, accuracy in accuracy_dict.items():
            axes[1].plot(x_axis, accuracy, label=f'{accuracy_type}: {max(accuracy):.3f}')
        axes[1].set(ylabel='Accuracy')
        axes[1].legend()
        enhance_plot(figure, axes[1], glow=True)
        figure.savefig(f'images/{file}.png')
        return figure, axes


def normalized_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
