import torch

from adversarial_training_box.pipeline.training_module import TrainingModule
from adversarial_training_box.database.experiment_tracker import ExperimentTracker
from adversarial_training_box.adversarial_attack.adversarial_attack import AdversarialAttack

class MixUpTrainingModule(TrainingModule):
    """ Code for mixup function from https://github.com/locuslab/robust_overfitting/tree/master """

    def __init__(self, criterion: torch.nn.Module, lambda_value: float) -> None:
        self.criterion = criterion
        self.lambda_value = lambda_value

    def mixup_data(self, data: torch.Tensor, target: torch.Tensor, lambda_value):

        batch_size = data.size()[0]
        index = torch.randperm(batch_size).cpu()

        mixed_data = lambda_value * data + (1 - lambda_value) * data[index, :]
        target_a, target_b = target, target[index]
        return mixed_data, target_a, target_b

    def mixup_criterion(self, criterion, output, target_a, target_b, lambda_value):
        return lambda_value * criterion(output, target_a) + (1 - lambda_value) * criterion(output, target_b)

    def train(self, data_loader: torch.utils.data.DataLoader, network: torch.nn.Module, optimizer: torch.optim, experiment_tracker: ExperimentTracker = None) -> float:
        if not experiment_tracker is None:
            experiment_tracker.watch(network, self.criterion, log_option="all", log_frequency=10)

        for batch_idx, (data, target) in enumerate(data_loader):

            mixed_data, target_a, target_b  = self.mixup_data(data, target, self.lambda_value)

            output = network(mixed_data)
            loss = self.mixup_criterion(self.criterion, output, target_a, target_b, self.lambda_value)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if not experiment_tracker is None:
                experiment_tracker.log({"loss": loss.item()})

        train_accuracy = (output.max(1)[1] == target).sum().item() / target.size(0)
        return train_accuracy


    def __str__(self) -> str:
        return f"standard_module_{self.criterion}_{self.lambda_value}"
    