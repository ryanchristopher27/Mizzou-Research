import torch
from mmpretrain import get_model

def main():
    model = get_model('convnext-tiny_32xb128_in1k', pretrained=True)
    inputs = torch.rand(1, 3, 224, 224)
    out = model(inputs)
    print(type(out))
    # To extract features.
    feats = model.extract_feat(inputs)
    print(type(feats))

if __name__ == '__main__':
    main()