# Imports
import torchvision.datasets as datasets
from torchvision.models import (
    ViT_B_16_Weights, 
    ResNet50_Weights
)
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

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

def get_k_fold_data(
    data_name: str = "ucmerced_landuse", 
    model_name: str = "vit_b_16", 
    num_folds: int = 5,
    fold: int = 1,
    train_batch_idx: int = 16,
    test_batch_size: int = 16,
    num_jobs,
    ):
        
    if model_name == "vit_b_16":
        transform = ViT_B_16_Weights.IMAGENETK_V1.transforms()
    
    if data_name == "ucmerced_landuse":
        dataset = datasets.ImageFolder(
            root = "Images",
            transform = transform,
        )
    
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    fold_idxs = list(kf.split(dataset))

    train_idx, test_idx = fold_idxs[fold - 1]

    train_loader = DataLoader(
        dataset=dataset, 
        batch_size=train_batch_size, 
        num_workers=num_jobs,
        sampler=torch.utils.data.SubsetRandomSampler(train_idx),
    )
    test_loader = DataLoader(
        dataset=dataset, 
        batch_size=test_batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(test_idx),
    )

    return train_loader, test_loader


            
