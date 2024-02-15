# Imports
from torchvision.models import (
    vit_b_16, 
    ViT_B_16_Weights, 
    resnet50, 
    ResNet50_Weights
)
from torch import nn

def get_model(
    model_name: str ="vit_b_16", 
    pretrain_weights: bool = True, 
    input_features: int = 768, 
    num_classes: int = 10
    ):
    
    if model_name == "vit_b_16":
        if pretrain_weights:
            model = vit_b_16(ViT_B_16_Weights.IMAGENETK_V1)
        else:
            model = vit_b_16()

        model.heads[0] = nn.Linear(input_features, num_classes)
        model.encoder.requires_grad_(False)

    elif model_name == "resnet50":
        if pretrain_weights:    
            model = resnet50(ResNet50_Weights.DEFAULT)
        else:
            model = resnet50()

        model.fc = nn.Linear(in_features=input_features, out_features=num_classes, bias=True)

    return model