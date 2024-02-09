# Imports
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.utils.data import DataLoader
from torch import nn
import torch
import os
from torchvision.io import read_image
from PIL import Image
from torchvision.transforms import ToTensor
import torchvision.datasets as datasets
from torch.utils.data import random_split
from torchvision.transforms import transforms
import numpy as np
from tqdm import tqdm
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
from sklearn.metrics import precision_recall_fscore_support as evaluate

from utils import *

# GLOBAL VARIABLES
TORCH_NUM_JOBS = int(os.environ.get("TORCH_NUM_JOBS", "4"))
TORCH_NUM_EPOCHS = int(os.environ.get("TORCH_NUM_EPOCHS", "20"))

# Function to preprocess images
def preprocess_images(images):
    tensors = []
    for img in tqdm(images, desc="Preprocessing"):
        tensors.append(torch.Tensor(img))
    return torch.cat(tensors, dim=0)

# DATA LOADER
def get_data(train_batch_size, test_batch_size) -> ():
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Resize((256, 256)),
    #     transforms.Lambda(lambda x: (torch.Tensor(x[1]).long(),
    #         torch.nn.functional.one_hot(torch.Tensor(x[1]).long(), num_classes=21))),
    #     # transforms.Lambda(lambda x: (preprocess_images(x[0]),
    #     #     torch.Tensor(x[1]).long(),
    #     #     torch.nn.functional.one_hot(torch.Tensor(x[1]).long(), num_classes=21))),
    # ])

    print(os.path.abspath('.'))
    total_dataset = datasets.ImageFolder('src/torch_code/Images', transform=ViT_B_16_Weights.IMAGENET1K_V1.transforms())
    # total_dataset = datasets.ImageFolder('Images', transform=ViT_B_16_Weights.IMAGENET1K_V1.transforms())
    # total_dataset = datasets.ImageFolder('src/torch_code/Images', transform=transform)

    train_size = int(0.8 * len(total_dataset))
    test_size = len(total_dataset) - train_size

    train_dataset, test_dataset = random_split(total_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, 
                                batch_size=train_batch_size, 
                                shuffle=True,
                                num_workers=TORCH_NUM_JOBS)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size)

    return train_loader, test_loader



# MAIN FUNCTION
def main():
    cuda_setup()

    train_batch_size = 16
    test_batch_size = 16

    train_data_loader, test_data_loader = get_data(train_batch_size=train_batch_size, test_batch_size=test_batch_size)

    # num_classes = train_data_loader.dataset.num_classes

    model = vit_b_16(ViT_B_16_Weights.IMAGENET1K_V1)

    # set output neurons to Num Classes = 21
    model.heads[0] = nn.Linear(768, 21)

    # freeze the backbone
    model.encoder.requires_grad_(False)

    # create opt and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_function = nn.CrossEntropyLoss()

    model.cuda()
    model.train()

    for epoch in range(TORCH_NUM_EPOCHS):
        print("===" * 30 + f"\nEpoch [{epoch+1} / {TORCH_NUM_EPOCHS}]")
        epoch_loss = 0
        epoch_correct = 0

        for i, (images, labels) in tqdm(enumerate(train_data_loader), total=len(train_data_loader)):
            images = images.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs.float(), labels.long())

            if torch.isnan(loss):
                raise RuntimeError("Loss reached NaN!")

            loss.backward()
            optimizer.step()

            _, predictions = torch.max(outputs, 1)
            epoch_correct += torch.sum(predictions == labels)
            epoch_loss += loss.item()

            # if i > 0 and (i % (len(train_data_loader) // 10) == 0 or i == 1):
            #     print(f"{i} / {len(train_data_loader)}" + 
            #         f"\tLoss = {epoch_loss / i:.2f}" + 
            #         f"\tAcc = {epoch_correct:d} / {i * train_data_loader.batch_size} " + 
            #         f"({epoch_correct / (i * train_data_loader.batch_size) * 100:.1f}%)", flush=True)

        print(f"Loss = {epoch_loss / len(train_data_loader):.4f}")
        print(f"Train Acc = {epoch_correct / (len(train_data_loader) * train_batch_size) * 100:.2f}%")
            

    model.eval()

    predictions = []
    labels = []
    with torch.no_grad():
        print("---" * 30 + f"\nRunning Eval")
        for i, (images, lb) in tqdm(enumerate(test_data_loader), total=len(test_data_loader)):

            model_outputs = model(images.cuda())

            _, preds = torch.max(model_outputs, 1)

            labels.extend(lb.numpy().tolist())
            predictions.extend(preds.cpu().numpy().tolist())
            # if i > 0 and i % (len(test_data_loader) // 10) == 0:
            #     print(f"{i} / {len(test_data_loader)}", flush=True)

    acc = metrics.accuracy_score(predictions, labels)
    prec, rec, fscore, _ = evaluate(predictions, labels, average="macro", zero_division=0)

    print("*" * 20 + f"""\n
    Accuracy  \t{acc*100:.2f}%
    Precision  \t{prec*100:.2f}%
    Recall  \t{rec*100:.2f}%
    F-1 Score \t{fscore*100:.2f}%
    """)

if __name__ == "__main__":
    main()