import wandb
import torch
from pathlib import Path
import pandas as pd
from datetime import datetime
from onnx2torch import convert
import json

from adversarial_training_box.database.attribute_dict import AttributeDict

class ExperimentTracker:
    def __init__(self, project: str, base_path: Path, login: bool) -> None:
        
        self.logged_in = login
        self.project = project
        self.project_path = base_path / project 
    
    def initialize_new_experiment(self, experiment_name: str, training_parameters: AttributeDict):
        now = datetime.now()
        now_string = now.strftime("%d-%m-%Y+%H_%M")
        run_string = f"{experiment_name}_{now_string}"
        self.training_parameters = training_parameters

        if self.logged_in:
            self.run = wandb.init(
            # set the wandb project where this run will be logged
            project=self.project,
            # track hyperparameters and run metadata
            config=self.training_parameters,
            name = run_string
            )

        self.experiment_name = run_string
        
        self.act_experiment_path = self.project_path / self.experiment_name

        if not self.act_experiment_path.exists():
            self.act_experiment_path.mkdir(parents=True)

        self.save_configuration(self.act_experiment_path, self.training_parameters)

    def save_configuration(self, path: Path, data: dict) -> None:
        data = data
        with open(path / "configuration.json", "w") as outfile: 
            json.dump(dict(data), outfile)

    def load(self, experiment_name: Path):
        self.act_experiment_path = self.project_path / experiment_name
    
    def load_trained_model(self, network: torch.nn.Module) -> torch.nn.Module:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_model = torch.load(self.act_experiment_path / f"{network.name}.pth", map_location=torch.device(device))
        torch_model.to(device)
        return torch_model
    
    def export_to_onnx(self, torch_model: torch.nn.Module, data_loader: torch.utils.data.DataLoader):
        torch_model.eval() 
        
        example_input, _ = next(iter(data_loader))
        example_input = example_input[0]

        device = next(torch_model.parameters()).device
        example_input = example_input.to(device)

        if ("cnn" in torch_model.name) or ("cifar" in torch_model.name) or ("resnet" in torch_model.name) or ("conv" in torch_model.name) or ("gtsrb" in torch_model.name):
            example_input = example_input.unsqueeze(0)
        torch.onnx.export(torch_model, example_input, 
                        self.act_experiment_path / f"{torch_model.name}.onnx",
                        export_params=True,
                        input_names = ['input'],
                        output_names = ['output'],
                        dynamic_axes={'input' : {0 : 'batch_size'},
                            'output' : {0 : 'batch_size'}},
                            opset_version=16)
    
    def log(self, information: dict) -> None:
        if self.logged_in:
            wandb.log(information)

    def save_model(self, network, upload_model=False) -> None:
        wandb.unwatch()
        model_path = self.act_experiment_path / f"{network.name}.pth"
        torch.save(network, model_path)
        
        if upload_model:
            artifact = wandb.Artifact('model', type='model')
            artifact.add_file(model_path)
            self.run.log_artifact(artifact)

    def watch(self, network, criterion, log_option, log_frequency):
        if self.logged_in:
            wandb.watch(network, criterion, log_option, log_frequency)

    def save_class_accuracy_table(self, data: list[tuple[int, float]]):
       columns = ["class", "accuracy",]
       table = wandb.Table(data=data, columns=columns)
       wandb.log({"class_wise_accuracy" : table})
    
    def log_train_accuracies(self, training_metric: dict): 
        """Save final train and validation accuracies to local file and to wandb"""
        result_df_path = self.act_experiment_path / "train_accuracies.csv"
        if result_df_path.exists():
            df = pd.read_csv(result_df_path, index_col=0)
            df.loc[len(df.index)] = training_metric
        else:
            df = pd.DataFrame([training_metric])
        df.to_csv(result_df_path)

        if self.logged_in:
            table = wandb.Table(dataframe=df)
            wandb.log({"train_accuracies" : table})

    def log_training_metrics(self, training_metric: dict): 
        """Save training time and other training metrics to local file and to wandb"""
        result_df_path = self.act_experiment_path / "training_metrics.csv"
        if result_df_path.exists():
            df = pd.read_csv(result_df_path, index_col=0)
            df.loc[len(df.index)] = training_metric
        else:
            df = pd.DataFrame([training_metric])
        df.to_csv(result_df_path)

        if self.logged_in:
            table = wandb.Table(dataframe=df)
            wandb.log({"metrics_table" : table})

    def log_test_result(self, result: dict):
        result_df_path = self.act_experiment_path / "accuracy_table.csv"
        if result_df_path.exists():
            df = pd.read_csv(result_df_path, index_col=0)
            df.loc[len(df.index)] = result
        else:
            df = pd.DataFrame([result])
        df.to_csv(result_df_path)

    def log_table_result_table_online(self):
        result_df_path = self.act_experiment_path / "accuracy_table.csv"
        df = pd.read_csv(result_df_path, index_col=0)

        if self.logged_in:
            table = wandb.Table(dataframe=df)
            wandb.log({"accuracy_table" : table})
        else:
            print("[WARNING] results can't be logged online. Login is set to false")