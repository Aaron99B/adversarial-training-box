import torch
import torchvision
import numpy as np
from torch import nn

from adversarial_training_box.adversarial_attack.adversarial_attack import AdversarialAttack

from cleverhans.torch.attacks.projected_gradient_descent import (
    fast_gradient_method
)

class FGSMAttack(AdversarialAttack):

    def __init__(self) -> None:
        super().__init__("FGSM")

    # FGSM attack code
    def fgsm_attack(self, image, epsilon, data_grad):
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = image + epsilon*sign_data_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        return perturbed_image

    # restores the tensors to their original scale
    def denorm(self, batch, mean=[0.1307], std=[0.3081]):
        """
        Convert a batch of tensors to their original scale.

        Args:
            batch (torch.Tensor): Batch of normalized tensors.
            mean (torch.Tensor or list): Mean used for normalization.
            std (torch.Tensor or list): Standard deviation used for normalization.

        Returns:
            torch.Tensor: batch of tensors without normalization applied to them.
        """
        if isinstance(mean, list):
            mean = torch.tensor(mean)
        if isinstance(std, list):
            std = torch.tensor(std)

        return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)


    def compute_perturbed_image(self, network: torch.nn.Module, data: torch.tensor, labels: torch.tensor, epsilon: float) -> torch.tensor:

        data.requires_grad = True

        output = network(data)

        loss_fn = nn.CrossEntropyLoss()

        loss = loss_fn(output, labels)

        network.zero_grad()

        loss.backward()

        data_grad = data.grad.data

        perturbed_data = self.fgsm_attack(data, epsilon, data_grad)

        return perturbed_data
    
    def __str__(self) -> str:
        return self.name