import torch

from adversarial_training_box.pipeline.training_module import TrainingModule
from adversarial_training_box.database.experiment_tracker import ExperimentTracker
from adversarial_training_box.adversarial_attack.adversarial_attack import AdversarialAttack

class StandardTrainingModule(TrainingModule):

    def __init__(self, criterion: torch.nn.Module, attack: AdversarialAttack = None, epsilon: float = None) -> None:
        self.criterion = criterion
        self.attack = attack
        self.epsilon = epsilon


    def train(self, data_loader: torch.utils.data.DataLoader, network: torch.nn.Module, optimizer: torch.optim, experiment_tracker: ExperimentTracker = None) -> float:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not experiment_tracker is None:
            experiment_tracker.watch(network, self.criterion, log_option="all", log_frequency=10)

        for batch_idx, (data, target) in enumerate(data_loader):
            
            data, target = data.to(device), target.to(device)

            if not self.attack is None:
                data = self.attack.compute_perturbed_image(network=network, data=data, labels=target, epsilon=self.epsilon)
                data.to(device)

            optimizer.zero_grad()
            output = network(data)
            loss = self.criterion(output, target)
            loss.backward()
            optimizer.step()

            if not experiment_tracker is None:
                experiment_tracker.log({"loss": loss.item()})

        train_accuracy = (output.max(1)[1] == target).sum().item() / target.size(0)
        return train_accuracy

    def __str__(self) -> str:
        return f"standard_module_{self.criterion}_{self.attack}_{self.epsilon}"
    