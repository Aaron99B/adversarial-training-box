from adversarial_training_box.pipeline.early_stopper import EarlyStopper
import torch
import time
import platform
import cpuinfo
from datetime import datetime
from collections import deque
import numpy as np

from adversarial_training_box.database.attribute_dict import AttributeDict
from adversarial_training_box.database.experiment_tracker import ExperimentTracker
from adversarial_training_box.pipeline.training_module import TrainingModule
from adversarial_training_box.pipeline.test_module import TestModule


class Pipeline:
    def __init__(self, experiment_tracker: ExperimentTracker, training_parameters: AttributeDict, criterion: torch.nn.Module, optimizer: torch.optim, scheduler: torch.optim=None) -> None:
        self.experiment_tracker = experiment_tracker
        self.training_parameters = training_parameters
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def save_model(self, network):
        self.experiment_tracker.save_model(network)

    def train(self, train_loader: torch.utils.data.DataLoader, network: torch.nn.Module, training_stack: list[int, TrainingModule], validation_module: TestModule = None, validation_loader: torch.utils.data.DataLoader = None, early_stopper : EarlyStopper = None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        network.to(device)

        training_starttime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            device_properties = torch.cuda.get_device_properties(0)
            total_memory_gb = device_properties.total_memory / (1024**3) 
            device_type = "GPU"
            device_info = f"{device_type}: {device_name} ({total_memory_gb:.1f} GB VRAM)"
        else:
            try:
                cpu_info = cpuinfo.get_cpu_info()
                cpu_name = cpu_info.get('brand_raw', 'Unknown CPU')
            except:
                cpu_name = platform.processor() or "Unknown CPU"
            device_type = "CPU"
            device_info = f"{device_type}: {cpu_name}"

        # Experiment metrics
        training_time = 0

        # Early stopper data
        best_validation_accuracy=-1.0
        early_stopping = False
        early_stopping_epoch = None

        if validation_module and hasattr(validation_module, 'attack') and validation_module.attack:
            smoothing_window_size = 10
            val_robust_acc_history = deque(maxlen=smoothing_window_size)

        # Create CUDA event objects for timing
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else: 
            start_time = time.perf_counter()


        for epochs, module in training_stack:
            for epoch in range(0, epochs):
                train_accuracy, train_robust_accuracy = module.train(train_loader, network, self.optimizer, self.experiment_tracker)
                validation_accuracy = train_accuracy

                if not self.experiment_tracker is None:
                    self.experiment_tracker.log({"train_accuracy" : train_accuracy, "train_robust_accuracy" : train_robust_accuracy, "epoch": epoch +1})

                if not self.scheduler is None:
                    self.scheduler.step()
                
                if validation_module:
                    network.eval()
                    _, _, validation_accuracy, validation_robust_accuracy, validation_loss  = validation_module.test(validation_loader, network)
                    network.zero_grad()
                    self.experiment_tracker.log({"validation_accuracy" : validation_accuracy, "validation_robust_accuracy" : validation_robust_accuracy, "validation_loss" : validation_loss})
                
                    # Early stopper works on robust accuracy in adversarial training case and train accuracy in conventional case
                    if early_stopper:
                        if validation_robust_accuracy is not None: # Early Stopper based on smoothed robust accuracy
                            val_robust_acc_history.append(validation_robust_accuracy)
                        
                            if len(val_robust_acc_history) == smoothing_window_size:
                                smoothed_val_robust_accuracy = np.mean(list(val_robust_acc_history))

                                if self.experiment_tracker:
                                    self.experiment_tracker.log({"smoothed_val_robust_accuracy": smoothed_val_robust_accuracy})

                                if early_stopper.early_stop(smoothed_val_robust_accuracy):
                                    early_stopping = True
                                    early_stopping_epoch = epoch + 1
                                    print(f"Early stopped at epoch: {early_stopping_epoch}")
                                    break
                        else: # Early stopping call based on validation accuracy
                            if early_stopper.early_stop(validation_accuracy):
                                early_stopping = True
                                early_stopping_epoch = epoch + 1
                                print(f"Early stopped at epoch: {early_stopping_epoch}")
                                break

                    if validation_accuracy > best_validation_accuracy:
                        best_validation_accuracy = validation_accuracy
                        print(f"New best model found at epoch {epoch+1} with validation accuracy: {best_validation_accuracy:.4f}. Saving model.")
                        self.save_model(network)
                else: 
                    self.save_model(network)
        
        if torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()
            duration_ms = start_event.elapsed_time(end_event)
            training_time = duration_ms / 1000
        else: 
            end_time = time.perf_counter()
            training_time = end_time - start_time
        
        if not self.experiment_tracker is None:
            self.experiment_tracker.log_train_accuracies({"Train Accuracy" : train_accuracy, "Train Robust Accuracy" : train_robust_accuracy, "Validation Accuracy" : validation_accuracy, "Validation Robust Accuracy" : validation_robust_accuracy})
            self.experiment_tracker.log_training_metrics({"training_time (s)" : training_time, "early_stopping" : bool(early_stopping), "early_stopping_epoch" : early_stopping_epoch, "training_start_datetime" : training_starttime, "device_info" : device_info})

    def test(self, network: torch.nn.Module, test_loader: torch.utils.data.DataLoader, testing_stack: list[TestModule]):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        network.to(device)
        
        network.eval()
        for module in testing_stack:
        
            print(f'testing for attack: {module.attack} and epsilon: {module.epsilon}')

            attack, epsilon, test_accuracy, robust_accuracy, validation_loss = module.test(test_loader, network)
            self.experiment_tracker.log_test_result({"epsilon" : epsilon, "attack" : str(attack), "accuracy" : test_accuracy, "robust_accuracy" : robust_accuracy})

        self.experiment_tracker.log_table_result_table_online()

    def test_accuracy_class_wise(self, network: torch.nn.Module):
        classes = [0,1,2,3,4,5,6,7,8,9]
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        # again no gradients needed
        with torch.no_grad():
            for data in self.test_loader:
                images, labels = data
                outputs = network(images)
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1

        class_accuracies = []
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            class_accuracies.append([classname, round(accuracy, 1)])

        self.experiment_tracker.save_class_accuracy_table(class_accuracies)