"""ResNet-18 regression head."""
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


def build(kind=None, shape=None, pretrained=False):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features,1)
    return model
