import wandb
import torch
from pathlib import Path
import pandas as pd
from datetime import datetime

from adversarial_training_box.database.attribute_dict import AttributeDict

class ExperimentTracker:
    def __init__(self, project: str, base_path: Path, training_parameters: AttributeDict, login: bool) -> None:
        self.training_parameters = training_parameters

        network_name = training_parameters.network
        now = datetime.now()
        now_string = now.strftime("%d-%m-%Y+%H_%M")
        run_string = f"{network_name}_{now_string}"
        if login:
            self.run = wandb.init(
            # set the wandb project where this run will be logged
            project=project,
            # track hyperparameters and run metadata
            config=training_parameters,
            name = run_string
            )

        self.experiment_name = run_string
        
        self.logged_in = login

        self.path = base_path / project / self.experiment_name
        if not self.path.exists():
         self.path.mkdir(parents=True)
    
    def log(self, information: dict) -> None:
        wandb.log(information)

    def save_model(self, network, data, upload_model=False) -> None:
        model_path = self.path / f"{network.name}.onnx"
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

    def log_test_result(self, result: dict):
        result_df_path = self.path / "accuracy_table.csv"
        if result_df_path.exists():
            df = pd.read_csv(result_df_path, index_col=0)
            df.loc[len(df.index)] = result
        else:
            df = pd.DataFrame([result])
        df.to_csv(result_df_path)

    def log_table_result_table_online(self):
        result_df_path = self.path / "accuracy_table.csv"
        df = pd.read_csv(result_df_path, index_col=0)

        if self.logged_in:
            table = wandb.Table(dataframe=df)
            wandb.log({"accuracy_table" : table})
        else:
            print("[WARNING] results can't be logged online. not logged in.")