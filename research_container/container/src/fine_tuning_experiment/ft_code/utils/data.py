# Imports
import torchvision.datasets as datasets
from torchvision.models import (
    ViT_B_16_Weights, 
    ViT_B_32_Weights,
    ViT_L_16_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
)
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import torch
from torchvision.transforms import ToTensor, transforms
from torch.utils.data import random_split


# Get Data For K-Fold Cross Validation
# Arguments
    # data_name: Name of the dataset
    # model_name: Name of the model
    # num_folds: Number of folds to split the data into
    # fold: Index of the fold to use (1 indexed)
    # train_batch_size: Batch size for training
    # test_batch_size: Batch size for testing
    # num_jobs: Number of jobs to run on torch

# Returns:
    # Train Data Loader
    # Test Data Loader

def get_kfold_data(
    data_name: str = "ucmerced_landuse", 
    model_name: str = "vit_b_16", 
    num_folds: int = 5,
    fold: int = 1,
    train_batch_size: int = 16,
    test_batch_size: int = 16,
    num_jobs: int = 4
    ):
        
    if model_name == "vit_b_16":
        transform = ViT_B_16_Weights.IMAGENET1K_V1.transforms()
        input_features = 768

    elif model_name == "vit_b_32":
        transform = ViT_B_32_Weights.IMAGENET1K_V1.transforms()
        input_features = 768

    elif model_name == "vit_l_16":
        transform = ViT_L_16_Weights.IMAGENET1K_V1.transforms()
        input_features = 1024

    elif model_name == "resnet18":
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_features = 512

    elif model_name == "resnet50":
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_features = 2048
    
    if data_name == "ucmerced_landuse":
        dataset = datasets.ImageFolder(
            root = "/rchristopher/data/src/data/UCMerced_Landuse",
            transform = transform,
        )
        num_classes = 21

    elif data_name == "cifar10":
        dataset = datasets.CIFAR10(
            root='data/', 
            download=True, 
            transform=transform
        )
        num_classes = 10
    
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    fold_idxs = list(kf.split(dataset))

    train_idx, test_idx = fold_idxs[fold - 1]

    print(f'Fold Number: {fold}')
    print(f'Train Indexes: {train_idx[0:10]}')
    print(f'Test Indexes: {test_idx[0:10]}')

    train_loader = DataLoader(
        dataset=dataset, 
        batch_size=train_batch_size, 
        num_workers=num_jobs,
        sampler=torch.utils.data.SubsetRandomSampler(train_idx),
        pin_memory=True,
    )
    test_loader = DataLoader(
        dataset=dataset, 
        batch_size=test_batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(test_idx),
    )

    return train_loader, test_loader, input_features, num_classes


def get_traintest_data(
    data_name: str = "cifar10", 
    model_name: str = "vit_b_16", 
    train_ratio: float = 0.8,
    train_batch_size: int = 16,
    test_batch_size: int = 16,
    num_workers: int = 4
    ):
        
    if model_name == "vit_b_16":
        transform = ViT_B_16_Weights.IMAGENET1K_V1.transforms()
        input_features = 768

    elif model_name == "vit_b_32":
        transform = ViT_B_32_Weights.IMAGENET1K_V1.transforms()
        input_features = 768

    elif model_name == "vit_l_16":
        transform = ViT_L_16_Weights.IMAGENET1K_V1.transforms()
        input_features = 1024

    elif model_name == "resnet18":
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_features = 512

    elif model_name == "resnet50":
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_features = 2048
    
    if data_name == "ucmerced_landuse":
        dataset = datasets.ImageFolder(
            root = "/rchristopher/data/src/data/UCMerced_Landuse",
            transform = transform,
        )
        num_classes = 21

        train_size = int(train_ratio * len(dataset))
        test_size = len(dataset) - train_size

        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


    elif data_name == "cifar10":
        train_dataset = datasets.CIFAR10(
            root='data/', 
            download=True, 
            transform=transform,
            train=True,
        )

        test_dataset = datasets.CIFAR10(
            root='data/', 
            download=True, 
            transform=transform,
            train=False,
        )

        num_classes = 10

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=train_batch_size, 
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=test_batch_size,
    )

    return train_loader, test_loader, input_features, num_classes
  
# DATA LOADER
def get_data(
    data_name: str = "ucmerced_landuse", 
    model_name: str = "vit_b_16", 
    num_folds: int = 5,
    fold: int = 1,
    train_ratio: float = 0.8,
    train_batch_size: int = 16,
    test_batch_size: int = 16,
    num_workers: int = 4
    ) -> tuple:

    if data_name == "cifar10":
        train_loader, test_loader, input_features, num_classes = get_traintest_data(
            data_name=data_name,
            model_name=model_name,
            train_ratio=train_ratio,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            num_workers=num_workers
        )
    elif data_name == "ucmerced_landuse":
        train_loader, test_loader, input_features, num_classes = get_kfold_data(
            data_name=data_name,
            model_name=model_name,
            num_folds=num_folds,
            fold=fold,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            num_jobs=num_workers
        )

    return train_loader, test_loader, input_features, num_classes