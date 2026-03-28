# Visualize if model is confounded
import torch
import numpy as np
from model.model import load_model
from dataset.dataset import load_data, create_dataloaders
from functions.optimizer import load_optimizer
from functions.loss import load_loss_fun
from functions.functions import train_model, eval_model, save_checkpoint, load_checkpoint
from functions.xai import explain_dataset, visualize_k_expl, evaluate_explainations
from utils.utils import enable_reproducibility

def see_explainations(model_name, dataset, bias_ratio,conf_type, seed=123):
  use_cuda = torch.cuda.is_available()
  device = 'cuda' if use_cuda else 'cpu'
  enable_reproducibility(seed)

  model = load_model(model_name, device=device)
  optim = load_optimizer("SGD", model.parameters(), lr=1e-2, weight_decay=0)
  loss = load_loss_fun("CrossEntropy")

  train_set, val_set, test_set = load_data(
    dataset, 
    seed=seed, 
    reload=True,
    bias_ratio=bias_ratio,
    variation=conf_type
  )
  
  data = [train_set, val_set, test_set]
  params = {"batch_size":32}
  m_params = [params]*3
  train_loader, val_loader, test_loader = create_dataloaders(data, m_params)

  _, _ = train_model(
    model=model, 
    train_loader=train_loader, 
    optimizer=optim, 
    loss_fun=loss, 
    n_epochs=10, 
    eval_loader=val_loader, 
    device=device
  )
  attrs, imgs = explain_dataset(train_loader, model, device)
  print(evaluate_explainations(attrs, train_set.masks, train_set.y)) 
  #for cls in range(10):
  #  visualize_k_expl(attrs, imgs, train_set, cls, 3)

bs1 = [0.99]*10
bs2 = [0,0,0,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99]
bs3 = [0,0,0,0,0,0,0,0.99,0.99,0.99]
BS_LIST = [
  [0.99]*10,
  [0.50]*10,
  [0,0,0,0.99,0.99,0.99,0.99,0.99,0.99,0.99],
  [0,0,0,0.5,0.5,0.5,0.5,0.5,0.5,0.5],
  [0,0,0,0,0,0.99,0.99,0.99,0.99,0.99],
  [0,0,0,0,0,0.5,0.5,0.5,0.5,0.5],
  [0,0,0,0,0,0,0,0.99,0.99,0.99],
  [0,0,0,0,0,0,0,0.5,0.5,0.5]
]

for bs in BS_LIST:
  print(f"Bias ratio {bs}")
  see_explainations(
    "ModernLeNet", 
    "DecoyFashionMNIST", 
    bs,
    2,
    123
    )