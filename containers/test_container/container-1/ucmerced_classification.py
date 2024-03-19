# Imports
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.io import read_image
from PIL import Image
from torchvision.transforms import ToTensor
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.transforms import transforms
import numpy as np
import torch
from torch import nn
from tqdm import tqdm as progressbar
from tqdm import trange
from skimage.feature import (
    local_binary_pattern as lbp,
    hog
)
from matplotlib import pyplot as plt
from torchvision.transforms.functional import to_tensor
import cv2 as cv
from sklearn import metrics
from torchvision.transforms.functional import normalize

def cuda_setup():
    if torch.cuda.is_available():
        print(torch.cuda.current_device())     # The ID of the current GPU.
        print(torch.cuda.get_device_name(id))  # The name of the specified GPU, where id is an integer.
        print(torch.cuda.device(id))           # The memory address of the specified GPU, where id is an integer.
        print(torch.cuda.device_count())
        
    on_gpu = torch.cuda.is_available()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    return device, on_gpu

def preprocess(image):
    # 28x28 is too small for VGG, the min size is 32x32
    image = cv.resize(image, (256, 256))
    # image = cv.cvtColor(image, cv.COLOR)
    # use torchvision.transforms.functional.to_tensor to convert it 
    tensor = to_tensor(image).unsqueeze(0)
    
    return tensor

def train(model, train_data, train_labels, loss_function, optimizer, normalize=True, batch_size=16, on_gpu=False, device="cpu"):
    
    model.train()
    
    # we'll keep track of the loss 
    # and acc of our model
    epoch_loss = 0
    epoch_correct = 0
    
    # the number of batches that we will need for an epoch 
    # of training
    n_batches = len(train_data) // batch_size
    
    # should we normalize our data
    if normalize:
        # subtract mean and divide by std
        train_data = (train_data - train_data.mean()) / train_data.std()
    
    # for each batch (with progress bar)
    for batch_idx in trange(n_batches):
        # get the start and end idx for the batch
        batch_start = batch_size * batch_idx
        batch_end = batch_start + batch_size
        
        # get the data and labels for this batch
        train_batch = train_data[batch_start:batch_end]
        labels = train_labels[batch_start:batch_end]

        if on_gpu:
            train_batch, labels = train_batch.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        model_outputs = model(train_batch)
        
        # calculate the loss
        # using the model outputs
        # and the desired labels
        _, labels_c = torch.max(labels, 1)
        
        # NLL and CrossEntropy need labels as [0, 4, 6, etc]
        if isinstance(loss_function, (torch.nn.NLLLoss, torch.nn.CrossEntropyLoss)): 
            loss = loss_function(model_outputs.float(), labels_c.long())
        # the others need 1 hot encoding
        else:
            loss = loss_function(model_outputs.float(), labels.float())
        
        # check that the loss is still real values
        if torch.isnan(loss):
            raise RuntimeError("Loss reached NaN!")
        
        loss.backward()
        optimizer.step()
        
        
        _, predictions = torch.max(model_outputs, 1)
        epoch_correct += torch.sum(predictions == labels_c)        
        
        # add this batch's loss to the total loss
        epoch_loss += loss.item()

    # print the avg loss and acc of all the batches
    # for this epoch
    print(f"Loss = {epoch_loss / n_batches:.4f}")
    print(f"Train Acc = {epoch_correct / len(train_labels) * 100:.2f}%") 
    
    # return the model
    return model

def main():

    device, on_gpu = cuda_setup()

    total_dataset = datasets.ImageFolder('Images/', transform=None)
    # total_dataset = datasets.ImageFolder('../datasets/UCMerced_LandUse/Images/', transform=None)
    train_size = int(0.8 * len(total_dataset))
    test_size = len(total_dataset) - train_size

    train_dataset, test_dataset = random_split(total_dataset, [train_size, test_size])

    # train_dataset_loader = DataLoader(dataset=train_dataset, batch_size=16)
    # test_dataset_loader = DataLoader(dataset=test_dataset, batch_size=16)

    # use list comprehension to convert every train image to np.ndarray
    train_images = [np.asarray(img) for img, _ in train_dataset]
    # iterate through the dataset again to get the labels  as a single array
    train_labels = [label for _, label in train_dataset]

    # repeat this same process for the test dataset
    test_images = [np.asarray(img) for img, _ in test_dataset]
    test_labels = [label for _, label in test_dataset]

    train_tensors = []
    for img in progressbar(train_images):
        train_tensors.append(preprocess(img))
    train_tensors = torch.cat(train_tensors, dim=0)

    test_tensors = []
    for img in progressbar(test_images):
        test_tensors.append(preprocess(img))
    test_tensors = torch.cat(test_tensors, dim=0)

    train_labels_tensor = torch.Tensor(train_labels).long()
    train_labels_tensor_1hot = torch.nn.functional.one_hot(train_labels_tensor, num_classes=21)

    test_labels_tensor = torch.Tensor(test_labels).long()
    test_labels_tensor_1hot = torch.nn.functional.one_hot(test_labels_tensor, num_classes=21)

    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)

    if on_gpu:
        print(f'Model on GPU')
        model.to(device)
    
    loss = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.0001)

    n_epochs = 10
    for i in range(n_epochs):
        print("---" * 20 + f"\nEpoch {i+1} / {n_epochs}")
        model = train(model, train_tensors, train_labels_tensor_1hot, loss, opt, batch_size=15, normalize=True, on_gpu=on_gpu, device=device)

    model.eval()
    # an empty tensor to hold predicted classes
    y_pred = np.empty(len(test_tensors))

    print("---" * 20)


    # this tells torch that we are only
    # going to perform forward passes
    with torch.no_grad():
        # for each feature
        for i, image in enumerate(progressbar(test_tensors)):
            if on_gpu:
                image = image.to(device)

            # forward pass - length of 10
            raw_prediction = model(image.unsqueeze(0))
            # get the argmax -- i.e., the cpredicted class
            _, predicted_class = torch.max(raw_prediction, 1)
            # save it to our list
            y_pred[i] = predicted_class.item()

    print(f'\nTest Accuracy: {metrics.accuracy_score(test_labels, y_pred)*100:.2f}%')


if __name__ == "__main__":
    main()