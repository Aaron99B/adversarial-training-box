import wandb
import torch
from pathlib import Path
import pandas as pd

from adversarial_training_box.database.attribute_dict import AttributeDict

class ExperimentTracker:
    def __init__(self, project: str, base_path: Path, training_parameters: AttributeDict) -> None:
        self.training_parameters = training_parameters
        self.run = wandb.init(
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

    def save_model(self, network, data, upload_model=False) -> None:
        model_path = self.path / "model.onnx"
        torch.onnx.export(network, data, model_path)
        
        if upload_model:
            artifact = wandb.Artifact('model', type='model')
            artifact.add_file(model_path)
            self.run.log_artifact(artifact)

    def watch(self, network, criterion, log_option, log_frequency):
       wandb.watch(network, criterion, log_option, log_frequency)

    def save_class_accuracy_table(self, data: list[tuple[int, float]]):
       columns = ["class", "accuracy",]
       table = wandb.Table(data=data, columns=columns)
       wandb.log({"class_wise_accuracy" : table})

    def log_table(self, name: str, data: list[dict]):
       df = pd.DataFrame.from_dict(data)
       table = wandb.Table(dataframe=df)
       wandb.log({name : table})