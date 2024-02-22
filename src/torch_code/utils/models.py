# Imports
from torchvision.models import (
    vit_b_16, 
    ViT_B_16_Weights, 
    vit_b_32,
    ViT_B_32_Weights,
    vit_l_16,
    ViT_L_16_Weights,
    resnet18,
    ResNet18_Weights,
    resnet50, 
    ResNet50_Weights,
)
from torch import nn
from mmpretrain import get_model
import torch

def get_kfold_model(
    model_name: str ="vit_b_16", 
    pretrain_weights: bool = True, 
    input_features: int = 768, 
    num_classes: int = 10
    ):
    
    if model_name == "vit_b_16":
        if pretrain_weights:
            model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            model = vit_b_16()

        model.heads[0] = nn.Linear(input_features, num_classes)
        model.encoder.requires_grad_(False)

    elif model_name == "vit_b_32":
        if pretrain_weights:
            model = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
        else:
            model = vit_b_32()

        model.heads[0] = nn.Linear(input_features, num_classes)
        model.encoder.requires_grad_(False)

    elif model_name == "vit_l_16":
        if pretrain_weights:
            model = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1)
        else:
            model = vit_l_16()

        model.heads[0] = nn.Linear(input_features, num_classes)
        model.encoder.requires_grad_(False)

    elif model_name == "resnet18":
        if pretrain_weights:
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            model = resnet18()

        model.fc = nn.Linear(input_features, num_classes, bias=True)

    elif model_name == "resnet50":
        if pretrain_weights:    
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            model = resnet50()

        model.fc = nn.Linear(input_features, num_classes, bias=True)

    elif model_name == "convnext-tiny_32xb128_in1k":
        if pretrain_weights:
            model = get_model('convnext-tiny_32xb128_in1k', pretrained=True, head=dict(num_classes=num_classes))
        else:
            model = get_model('convnext-tiny_32xb128_in1k', pretrained=False, head=dict(num_classes=num_classes))

        model.head.fc.weight = torch.Size([num_classes, input_features])
        model.head.fc.bias = torch.Size([num_classes])
        # model.fc = nn.Linear(input_features, num_classes, bias=True)

    return model