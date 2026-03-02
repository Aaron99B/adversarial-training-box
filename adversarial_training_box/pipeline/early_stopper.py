import torch

from adversarial_training_box.pipeline.standard_test_module import StandardTestModule
class EarlyStopper:
    def __init__(self, patience: int = 1, delta: float=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.max_validation_accuracy = 0

    def early_stop(self, validation_accuracy):
        if validation_accuracy > (self.max_validation_accuracy + self.delta):
            self.max_validation_accuracy = validation_accuracy
            self.counter = 0
        else:
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False