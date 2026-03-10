import torch
import numpy as np
from model.lenet import load_lenet
from dataset.dataset import load_data, create_dataloaders
from functions.optimizer import load_optimizer
from functions.loss import load_loss_fun
from functions.functions import train_model, eval_model, save_checkpoint, load_checkpoint
from functions.xil import compute_simplicity
from utils.utils import enable_reproducibility
from experiments.utils import compute_correlations

def exp_train_dynamics(
  seed: int = 123, 
  dataset: str = "DecoyMNIST", 
  variation: int = 0 ,
  metric: str= "MP") -> dict:
  """Run experiment to see how training dynamics correlate with confounder presence.
  Args:
    seed (int): seed for the experiment.
    dataset (str): dataset to use for the experiment.
    metric (str): metric to use for simplicity.
  Returns:
    dict: correlation between confounder presence and training dynamics.
  """
  use_cuda = torch.cuda.is_available()
  device = 'cuda' if use_cuda else 'cpu'

  enable_reproducibility(seed)
  model = load_lenet(device)
  optim = load_optimizer("SGD", model.parameters(), lr=1e-2, weight_decay=0)
  loss = load_loss_fun("CrossEntropy")

  train_set, val_set, test_set = load_data(
    dataset, 
    seed=seed, 
    reload=True,
    bias_ratio=[0.99]*10,
    variation=variation
  )
  
  data = [train_set, val_set, test_set]
  params = {"batch_size":32}
  m_params = [params]*3
  train_loader, val_loader, test_loader = create_dataloaders(data, m_params)

  _, dyn = train_model(
    model, 
    train_loader, 
    optim, 
    loss, 
    n_epochs=20, 
    eval_loader=val_loader, 
    device=device
  )
  #loss, acc = eval_model(model, test_loader, loss,  device)
  #print("="*20,f"Test set Loss:{loss:.2f} | Acc:{acc:.2f}.","="*20)
  
  simplicity = compute_simplicity(dyn, metric)

  separation_list = []
  is_confounded = []
  labels = []
  for id in range(len(train_set)):
    index, _, label, mask = train_set[id] 
    separation_list.append(simplicity[index])
    is_confounded.append(1 if mask.sum() > 1 else 0)
    labels.append(label.item())

  result = compute_correlations(separation_list, is_confounded, labels)

  return result