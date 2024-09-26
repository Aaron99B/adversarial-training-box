# Adversarial Training Box

-----
This Adversarial Training Box package simplifies your experiment pipeline for training neural networks using adversarial training. The entire package os class-based and can be easily extended with new training methods. The training progress can be logged via [Weights & Biases](https://wandb.ai/site)

## Installation

- clone the repository locally
- create new environment ```conda create -n adversarial-training-box python=3.10```
- activate the environment ```conda activate adversarial-training-box```
- change into verona directory ```cd adversarial_training_box```
- install dependencies ```pip install -r requirements.txt```
- install package locally (editable install for development) ```pip install -e .```

## Available Training Techniques
- Standard training
- FGSM
- PGD

## Tutorial 

### Example Scripts
There are a few example scripts in the ```example scripts``` folder to see, how one can do an HPO and adversarial training for different networks and datasets.


## Testing
The package was tested on the following datasets: MNIST, CIFAR-10 and GTSRB