# Imports
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import ToTensor
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from matplotlib import pyplot as plt
from torchvision.transforms.functional import to_tensor
import cv2 as cv
from sklearn import metrics
import torch.nn.functional as F

# Links
    # https://medium.com/@sergioalves94/deep-learning-in-pytorch-with-cifar-10-dataset-858b504a6b54

def main():
    dataset = datasets.CIFAR10(root='data/', download=True, transform=ToTensor())
    test_dataset = datasets.CIFAR10(root='data/', train=False, transform=ToTensor())

    num_classes = len(dataset.classes)

    torch.manual_seed(43)
    val_size = 5000
    train_size = len(dataset) - val_size
    test_size = len(test_dataset)

    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    len(train_ds), len(val_ds)

    batch_size = 128


    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size*2, num_workers=4, pin_memory=True)

    # Declare ResNet50 Model
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)

    model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)

    # Hyper Parameters
    n_epochs = 20
    lr = 0.0001

    # Loss Function and Optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training Analysis Variables
    train_loss = np.zeros(n_epochs)
    train_acc = np.zeros(n_epochs)
    val_loss = np.zeros(n_epochs)
    val_acc = np.zeros(n_epochs)

    # Train
    for epoch in range(n_epochs):
        # Print Epoch Separator
        print('==='*30)
        print(f'Epoch: [{epoch+1}/{n_epochs}]')

        n_batches_train = len(train_loader)
        train_loss_accumulator = 0
        train_correct_accumulator = 0

        for train_batch_idx, (features, labels) in tqdm(enumerate(train_loader), desc='Train', total=n_batches_train):
            
            # labels_one_hot = F.one_hot(labels, num_classes)

            optimizer.zero_grad()
            outputs = model(features)
            outputs_c = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            correct = (outputs_c.indices == labels).sum().item()

            loss.backward()
            optimizer.step()

            train_loss_accumulator += loss.item()
            train_correct_accumulator += correct

        ave_train_loss = train_loss_accumulator / n_batches_train
        epoch_train_acc = train_correct_accumulator / train_size

        print(f'Average Batch Train Loss: {ave_train_loss:.4}, Train Accuracy: {(epoch_train_acc * 100):.4}%\n')

        n_batches_val = len(val_loader)
        val_loss_accumulator = 0
        val_correct_accumulator = 0

        with torch.no_grad():
            for val_batch_idx, (features, labels) in tqdm(enumerate(val_loader), desc='Validation', total=n_batches_val):
                
                # labels_one_hot = F.one_hot(labels, num_classes)
                
                outputs = model(features)
                outputs_c = torch.max(outputs, 1)

                loss = criterion(outputs, labels)

                correct = (outputs_c.indices == labels).sum().item()

                val_loss_accumulator += loss.item()
                val_correct_accumulator += correct


        ave_val_loss = val_loss_accumulator / n_batches_val
        epoch_val_acc = val_correct_accumulator / val_size

        train_loss[epoch] = ave_train_loss
        train_acc[epoch] = epoch_train_acc
        val_loss[epoch] = ave_val_loss
        val_acc[epoch] = epoch_val_acc

        print(f'Average Batch Validation Loss: {ave_val_loss:.4}, Validation Accuracy: {(epoch_val_acc * 100):.4}%\n')

    # Testing
    n_batches_test = len(val_loader)
    test_loss_accumulator = 0
    test_correct_accumulator = 0

    with torch.no_grad():
        for test_batch_idx, (features, labels) in tqdm(enumerate(test_loader), desc='Test', total=n_batches_test):
            
            # labels_one_hot = F.one_hot(labels, num_classes)
            
            outputs = model(features)
            outputs_c = torch.max(outputs, 1)

            loss = criterion(outputs, labels)

            correct = (outputs_c.indices == labels).sum().item()

            test_loss_accumulator += loss.item()
            test_correct_accumulator += correct

        ave_test_loss = test_loss_accumulator / n_batches_test
        epoch_test_acc = test_correct_accumulator / test_size

        print('==='*30)
        print(f'\nAverage Batch Test Loss: {ave_test_loss:.4}, Test Accuracy: {(epoch_test_acc * 100):.4}%\n')
        print('==='*30)

    plt.plot(train_loss, marker='o', linestyle='-', color='b', label='Training Loss')
    plt.plot(val_loss, marker='x', color='r', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.plot(train_acc, marker='o', linestyle='-', color='b', label='Training Accuracy')
    plt.plot(val_acc, marker='x', color='r', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Epoch')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()