import torch
from mmpretrain import get_model
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor, transforms
from torch.utils.data import random_split
from torch import nn

def main():
    model = get_model('convnext-tiny_32xb128_in1k', pretrained=True, head=dict(num_classes=10))


    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.CIFAR10(
        root='data/', 
        download=True, 
        transform=transform
    )

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=32, 
        num_workers=4,
    )
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=32,
    )
    
    input_features = 768
    num_classes = 10

    model.data_preprocessor.num_classes = 10
    model.fc = nn.Linear(input_features, num_classes, bias=True)


    inputs = torch.rand(1, 3, 224, 224)
    out = model(inputs)
    print(type(out))
    # To extract features.
    feats = model.extract_feat(inputs)
    print(type(feats))

if __name__ == '__main__':
    main()