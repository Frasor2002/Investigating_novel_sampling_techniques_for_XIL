import torch
from model.lenet import load_lenet
from dataset.dataset import load_data, create_dataloaders
from functions.optimizer import load_optimizer
from functions.loss import load_loss_fun
from functions.functions import train_model, eval_model, save_checkpoint
from functions.xil import reset_model
from utils.utils import enable_reproducibility
from experiments.training_dynamic import exp_train_dynamics
from experiments.model_output import exp_model_outputs
from experiments.explaination_entropy import exp_explaination_entropy

# Experiment seed
SEED = 123

DATASET_NAMES = ["DecoyMNIST", "DecoyFashionMNIST"]

# Params
DATASET = DATASET_NAMES[0]

	
if __name__ == "__main__":
  use_cuda = torch.cuda.is_available()
  device = 'cuda' if use_cuda else 'cpu'

  res1 = exp_train_dynamics(device, seed=SEED)
  print(res1)

  res2 = exp_model_outputs(device, seed=SEED)
  print(res2)

  res3 = exp_explaination_entropy(device, seed=SEED)
  print(res3)