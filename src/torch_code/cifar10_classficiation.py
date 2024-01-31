from torchvision.models import resnet50, ResNet50_Weights
from torchvision.io import read_image
from PIL import Image
from torchvision.transforms import ToTensor
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from torchvision.transforms import transforms
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from skimage.feature import (
    local_binary_pattern as lbp,
    hog
)
from matplotlib import pyplot as plt
from torchvision.transforms.functional import to_tensor
import cv2 as cv
from sklearn import metrics
import torch.nn.functional as F

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
    n_epochs = 10
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

        print(f'Train Loss: {ave_train_loss}, Train Accuracy: {epoch_train_acc}')

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

        print(f'Validation Loss: {ave_val_loss}, Validation Accuracy: {epoch_val_acc}\n')



if __name__ == "__main__":
    main()