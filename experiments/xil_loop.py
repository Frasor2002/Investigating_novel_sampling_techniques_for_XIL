import torch
from model.model import load_model
from dataset.dataset import load_data, create_dataloaders
from functions.optimizer import load_optimizer
from functions.loss import load_loss_fun
from functions.functions import train_model, eval_model, save_checkpoint, load_checkpoint
from functions.xil import xil_loop, random_sampling, simplicity_sampling
from utils.utils import enable_reproducibility
from typing import Any

# Name of checkpoint to reset the model
RESET_CHECKPOINT="reset_model"

def exp_xil_loop(
  seed: int = 123, 
  model_name : str = "LeNet",
  model_variant: str = "modern",
  dataset: str = "DecoyMNIST", 
  variation: int = 0,
  sampling_strategy: Any = random_sampling,
  budget:int=1000,
  step:int=1,
  initial_query:int=0,
  rr_reg: float=1,
  log_filename:str="xil_log"
):
  use_cuda = torch.cuda.is_available()
  device = 'cuda' if use_cuda else 'cpu'
  enable_reproducibility(seed)

  if model_name == "LeNet":
    model = load_model(model_name, device=device, variant=model_variant)
  else:
    model = load_model(model_name, device=device)

  # setup for the reset checkpoint later
  save_checkpoint(RESET_CHECKPOINT, model)
  
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
    n_epochs=10, 
    eval_loader=val_loader, 
    device=device
  )
  loss, acc = eval_model(model, test_loader, loss,  device)
  print("="*20,f"Test set Loss:{loss:.2f} | Acc:{acc:.2f}.","="*20)

  #rr_reg = 1 if dataset == "DecoyMNIST" else 1e-3

  # Run XIL loop
  query = xil_loop(
    train_data=train_set,
    model=model, 
    sampling_strategy=sampling_strategy,
    budget=budget,
    val_loader=val_loader,
    test_loader=test_loader,
    tr_dynamics=dyn,
    step_size=step,
    starting_query=initial_query,
    rrr_reg_rate=rr_reg,
    log_filename=log_filename,
    device=device
  )






  
  
  