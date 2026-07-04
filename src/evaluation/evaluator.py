import torch

all_predictions = []
all_labels = []


@torch.no_grad
def evaluate(model, dataloader, device):
    """Run inference to the test set"""

    model.eval()

    for images, labels in dataloader:

        images = images.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.cpu().numpy())

        all_labels.extend(labels.numpy())

    return all_predictions, all_labels
