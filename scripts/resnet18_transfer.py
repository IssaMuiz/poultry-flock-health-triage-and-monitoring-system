from src.models.resnet18_transfer import ResNet18Transfer

model = ResNet18Transfer()

for name, parameter in model.named_parameters():
    if parameter.requires_grad:
        print(name)
