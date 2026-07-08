import torch


@torch.no_grad()
def collect_misclassified_images(model, dataloader, device):
    """collect all misclassified samples in the test dataset"""

    model.eval()

    mistakes = []

    for images, labels, paths in dataloader:

        images = images.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1).cpu()

        for image, label, prediction, path in zip(
            images.cpu(), labels, predictions, paths
        ):

            if label != prediction:
                mistakes.append(
                    {
                        "image": image,
                        "actual_label": int(label),
                        "predicted_label": int(prediction),
                        "path": path,
                    }
                )
    return mistakes
