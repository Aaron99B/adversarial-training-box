import torch

from adversarial_training_box.pipeline.training_module import TrainingModule
from adversarial_training_box.database.experiment_tracker import ExperimentTracker
from adversarial_training_box.adversarial_attack.adversarial_attack import AdversarialAttack

class StandardTrainingModule(TrainingModule):

    def __init__(self, criterion: torch.nn.Module, attack: AdversarialAttack = None, epsilon: float = None) -> None:
        self.criterion = criterion
        self.attack = attack
        self.epsilon = epsilon
        self.max_grad_norm = 1.0


    def train(self, data_loader: torch.utils.data.DataLoader, network: torch.nn.Module, optimizer: torch.optim, experiment_tracker: ExperimentTracker = None) -> float:
        network.train()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not experiment_tracker is None:
            experiment_tracker.watch(network, self.criterion, log_option="all", log_frequency=10)

        correct_predictions_adversarial = 0
        correct_predictions_benign = 0
        total_samples = 0

        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            training_data = data

            # In adversarial case, data is perturbed
            if not self.attack is None:
                training_data = self.attack.compute_perturbed_image(network=network, data=data, labels=target, epsilon=self.epsilon)

            # Forward pass
            output = network(training_data)
            loss = self.criterion(output, target)

            # Check for NaN/Inf loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss detected at batch {batch_idx}: {loss.item()}")
                continue

            # Backward pass
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(network.parameters(), self.max_grad_norm)

            # Update model
            optimizer.step()

            if not experiment_tracker is None:
                experiment_tracker.log({"train_loss": loss.item()})

            # Accumulate predictions for adversarial and benign case
            if not self.attack is None:
                predictions = output.max(1)[1]
                correct_predictions_adversarial += (predictions == target).sum().item()
                with torch.no_grad():
                    conventional_output = network(data)
                    predictions = conventional_output.max(1)[1]
                    correct_predictions_benign += (predictions == target).sum().item()
            else: 
                predictions = output.max(1)[1]
                correct_predictions_benign += (predictions == target).sum().item()

            total_samples += target.size(0)

        # Calculate accuracies
        train_accuracy = correct_predictions_benign/total_samples
        robust_accuracy = None
        if not self.attack is None:
            robust_accuracy = correct_predictions_adversarial / total_samples
        
        return train_accuracy, robust_accuracy

    def __str__(self) -> str:
        return f"standard_module_{self.criterion}_{self.attack}_{self.epsilon}"
    