import torch


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    """

    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        predictions = outputs.argmax(dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_samples += labels.size(0)

        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = correct_predictions / total_samples

    return epoch_loss, epoch_accuracy


@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device):
    """
    validate the model for one epoch.
    """
    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in dataloader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        running_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        correct_predictions += (predictions == labels).sum().item()

        total_samples += labels.size(0)

        epoch_loss = running_loss / len(dataloader)

        epoch_accuracy = correct_predictions / total_samples

    return epoch_loss, epoch_accuracy
