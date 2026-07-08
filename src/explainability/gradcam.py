import torch.nn.functional as F


class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self._register_hooks()

    def _register_hooks(self):

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):

            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)

        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, image_tensor):

        self.model.zero_grad()

        outputs = self.model(image_tensor)

        predictions = outputs.argmax(dim=1).item()

        score = outputs[0, predictions]

        score.backward()

        gradients = self.gradients

        activations = self.activations

        weights = gradients.mean(dim=(2, 3), keepdim=True)

        cam = (weights * activations).sum(dim=1)

        cam = F.relu(cam)

        cam = cam.squeeze()

        cam -= cam.min()

        cam /= cam.max() + 1e-8

        return cam.cpu().numpy(), predictions
