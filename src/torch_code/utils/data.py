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
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_features = 2048

    elif model_name == "convnext-tiny_32xb128_in1k":
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_features = 2048
    
    if data_name == "ucmerced_landuse":
        dataset = datasets.ImageFolder(
            root = "Images",
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

  
# DATA LOADER
def get_data(train_batch_size, test_batch_size) -> tuple:

    # print(os.path.abspath('.'))
    # total_dataset = datasets.ImageFolder('src/torch_code/Images', transform=ViT_B_16_Weights.IMAGENET1K_V1.transforms())
    total_dataset = datasets.ImageFolder('Images', transform=ViT_B_16_Weights.IMAGENET1K_V1.transforms())

    train_size = int(0.8 * len(total_dataset))
    test_size = len(total_dataset) - train_size

    train_dataset, test_dataset = random_split(total_dataset, [train_size, test_size])

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=train_batch_size, 
        num_workers=4,
    )
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=test_batch_size,
    )

    return train_loader, test_loader