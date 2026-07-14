import torch.nn as nn
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights


class ResNet18Transfer(nn.Module):
    """Transfer model learning using pretrained Resnet18"""

    def __init__(self, num_classes=2):
        super().__init__()

        # Load pretrained resnet18 model
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Replace the resnet18 final classifier
        self.model.fc = nn.Linear(
            in_features=self.model.fc.in_features, out_features=num_classes
        )

        # Freeze all pretrained layers
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        # Unfreeze classifier
        for parameter in self.model.fc.parameters():
            parameter.requires_grad = True

    def forward(self, x):
        return self.model(x)
