# Imports
from torchvision.models import (
    vit_b_16, 
    ViT_B_16_Weights, 
    resnet50, 
    ResNet50_Weights
)

def get_model(model_name: str ="vit_b_16", pretrain_weights: bool = True):
    if model_name == "vit_b_16":
        if pretrain_weights:
            model = vit_b_16(ViT_B_16_Weights.IMAGENETK_V1)
        else:
            model = vit_b_16()
    elif model_name == "resnet50":
        if pretrain_weights:    
            model = resnet50(ResNet50_Weights.DEFAULT)
        else:
            model = resnet50()

    return model