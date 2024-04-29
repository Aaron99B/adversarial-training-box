import wandb
import torch
from pathlib import Path

from adversarial_training_box.attribute_dict import AttributeDict

class ExperimentTracker:
    def __init__(self, project: str, base_path: Path, training_parameters: AttributeDict) -> None:
        self.training_parameters = training_parameters
        wandb.init(
        # set the wandb project where this run will be logged
        project=project,
        # track hyperparameters and run metadata
        config=training_parameters
        )

        self.experiment_name = wandb.run.name

        self.path = base_path / self.experiment_name
        if not self.path.exists():
         self.path.mkdir(parents=True)
    
    def log(self, information: dict) -> None:
        wandb.log(information)

    def save_model(self, network, data) -> None:
        torch.onnx.export(network, data, self.path / "model.onnx")
        wandb.run.save(self.path / "model.onnx")

    def watch(self, network, criterion, log_option, log_frequency):
       wandb.watch(network, criterion, log_option, log_frequency)