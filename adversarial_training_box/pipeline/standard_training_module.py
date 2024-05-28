import torch
from tqdm import tqdm

from adversarial_training_box.pipeline.training_module import TrainingModule
from adversarial_training_box.database.experiment_tracker import ExperimentTracker
from adversarial_training_box.adversarial_attack.adversarial_attack import AdversarialAttack

class StandardTrainingModule(TrainingModule):

    def __init__(self, number_epochs: int, criterion: torch.nn.Module, attack: AdversarialAttack | None, epsilon: float) -> None:
        self.number_epochs = number_epochs
        self.criterion = criterion
        self.attack = attack
        self.epsilon = epsilon


    def train(self, data_loader: torch.utils.data.DataLoader, network: torch.nn.Module, optimizer: torch.optim, experiment_tracker: ExperimentTracker = None) -> None:
        if not experiment_tracker is None:
            experiment_tracker.watch(network, self.criterion, log_option="all", log_frequency=10)

        for epoch in range(0,self.number_epochs):
            for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
                data = data.reshape(-1, 784)

                if not self.attack is None:
                    data = self.attack.compute_perturbed_image(network=network, data=data, labels=target, epsilon=self.epsilon)

                output = network(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if not experiment_tracker is None:
                    experiment_tracker.log({"loss": loss.item()})


    def __str__(self) -> str:
        return f"standard_module_{self.number_epochs}_{self.criterion}_{self.attack}_{self.epsilon}"
    