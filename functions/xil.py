from torch import nn
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from functions.functions import load_checkpoint, eval_model, train_model
from functions.loss import load_loss_fun
from functions.optimizer import load_optimizer
from functions.xai import explain_dataset, evaluate_explainations
from typing import Any, Callable, Optional
from tqdm import tqdm
import random
import numpy as np
import logging
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "log", "xil")

# Name of checkpoint to reset the model
RESET_CHECKPOINT="reset_model"

class XIL_Dataset(Dataset):
  def __init__(self, dataset: Any) -> None:
    super().__init__()
    self.dataset = dataset
    self.requested_ids = set()
  
  def activate_explanation(self, pos):
    sample_index, _, _, _ = self.dataset[pos]
    self.requested_ids.add(sample_index)
  
  def __getitem__(self, idx): 
    unique_id, x, y, real_mask = self.dataset[idx]
    if unique_id in self.requested_ids:
      mask_out = real_mask
    else:
      mask_out = torch.zeros_like(real_mask)
            
    return unique_id, x, y, mask_out

  def __len__(self):
    return len(self.dataset)

  def __getattr__(self, name):
    return getattr(self.dataset, name)




def xil_loop(
    train_data: Any, 
    model: nn.Module, 
    sampling_strategy: Callable,
    budget: int, 
    val_loader:DataLoader, 
    test_loader: DataLoader,
    tr_dynamics:Optional[dict]=None,
    step_size:int=1,
    starting_query:int=0,
    rrr_reg_rate:float=1, 
    log_filename: str= "xil_log",
    device:str="cpu") -> dict:
  """XIL loop to deconfound a model.
  Args:
    train_data (Any): traning dataset.
    model (Module): model that needs to be deconfounded.
    sampling_strategy (Callable): sampling strategy.
    budget (int): number of queries to do.
    test_loader(DataLoader): test DataLoader for evaluation.
    device (str): device where training happens.
  """

  log = {
    "accuracy": [],
    "query": []
  }

  # logging configuration
  os.makedirs(LOG_DIR, exist_ok=True)
  logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
      logging.FileHandler(os.path.join(LOG_DIR,f"{log_filename}.log")),
      logging.StreamHandler()
    ]
  )
  logger = logging.getLogger(__name__)

  xil_train_dataset = XIL_Dataset(train_data)

  all_positional_ids = list(range(len(train_data)))
  explained_ids = []

  if starting_query > 0:

    logger.info(f"Starting XIL loop with {starting_query} initial queries.")

    chosen_positions = sampling_strategy(
      all_positional_ids, 
      training_dynamics=tr_dynamics, 
      dataset=train_data, 
      k=starting_query)

    # SANITY CHECK
    chosen_labels = []
    from collections import Counter
    for pos in chosen_positions:
      # Unpack your dataset tuple (unique_id, x, y, mask)
      _, _, y, _ = train_data[pos] 
      # Handle both PyTorch tensors and standard ints
      label = y.item() if isinstance(y, torch.Tensor) else y
      chosen_labels.append(label)
        
    label_counts = Counter(chosen_labels)
    
    # Print it out nicely in your logger
    logger.info(f"--- Class Distribution for this {len(chosen_positions)} sample batch ---")
    for label, count in sorted(label_counts.items()):
      logger.info(f"  Class {label}: {count} samples")
    logger.info("---------------------------------------------------------")
    # SANITY CHECK
    
    for pos in chosen_positions:
      explained_ids.append(pos)
      xil_train_dataset.activate_explanation(pos)

  query_count = len(explained_ids)
  loop = tqdm(range(budget),initial=starting_query, desc="XIL Loop")

  while query_count in loop:

    sampling_pool = list(set(all_positional_ids) - set(explained_ids))
    if len(sampling_pool) == 0:
      logger.warning("Full explanatory supervision reached earlier than budget.")
      break

    # Step size must not exceed budget
    current_step = min(step_size, budget - query_count, len(sampling_pool))

    chosen_positions = sampling_strategy(sampling_pool, training_dynamics=tr_dynamics, dataset=train_data, k=current_step)
    
    # Process the batch
    for pos in chosen_positions:
      explained_ids.append(pos)
      xil_train_dataset.activate_explanation(pos)

    loop.set_postfix_str(f"{len(explained_ids)}/{budget} explained")
    query_count += len(chosen_positions)
    loop.update(len(chosen_positions))
    
    # Reset model and than retrain
    load_checkpoint(RESET_CHECKPOINT, model, device)
    # Init optimizer and losses
    optim = load_optimizer("SGD", model.parameters(), lr=1e-2, weight_decay=0)
    train_loss = load_loss_fun("RRR", reg_rate=rrr_reg_rate, rr_clip=2) # RRR
    eval_loss = load_loss_fun("CrossEntropy")
    train_loader = DataLoader(xil_train_dataset, batch_size=32)

    _, _ = train_model(model, train_loader, optim, train_loss, 10, val_loader, device=device)
    loss, acc = eval_model(model, test_loader, eval_loss, device)
    logger.info(f"Iteration Progress: {query_count}/{budget} samples explained. Performance acc= {acc}, loss={loss}")
    
    log['accuracy'].append(acc)
    log['query'].append(query_count)

    #all_attr, all_imgs = explain_dataset(val_loader, model, device)
    #exp_mse = evaluate_explainations(all_attr, val_loader.dataset.masks)
    #print(f"Explaination MSE: {exp_mse:.2f}")

  return log

def random_sampling(sampling_pool:list, k:int, **kwargs) -> list:
  """Baseline sampling strategy that picks k random samples from the pool.
  Args:
    sampling_pool (list): list of available sample positions.
    k (int): amount of samples.
  Returns:
    int: chosen samples.
  """
  chosen = random.sample(sampling_pool,k)
  return chosen

def simplicity_sampling(sampling_pool:list,training_dynamics: dict, dataset:Any, k:int) -> list:
  """Sample k samples by taking simplest ones.
  Args:
    sampling_pool (list): list of available sample positions.
    k (int): amount of samples.
    dataset (Any): dataset to map between dataset indeces and positional ones.
    training_dynamics (dict): training dynamics for each sample.
  Returns:
    int: chosen samples.
  """
  # sampling_pool uses positional ids, i need to return from ids of sample to positional
  simplicity = compute_simplicity(training_dynamics, metric = "MP") # ids of samples
  # Sort the sampling pool by simplicity
  sorted_pool = sorted(
    sampling_pool, 
    key=lambda internal_idx: simplicity[dataset.indices[internal_idx]], 
    reverse=True
  )

  chosen = sorted_pool[:k]

  return chosen


def compute_simplicity(training_dynamics: dict, metric: str = "MP") -> dict:
  """For every sample convert the traning dynamics into simplicity metric.
  The idea is that samples need to be ranked based on when they are learned.
  This is because samples that contain confounders are more simple to learn and therefore are
  learned faster.
  Args:
    training_dynamics (dict): dynamics for each sample during training.
    metric (str): metric to be used "MP" or "EC".
  Returns:
    dict: simplicity dict mapping id -> simplicity score.
  """
  if metric not in ["MP", "EC"]:
    raise ValueError("Not valid simplicity metric.")

  simplicity = {}

  for id, epoch_metrics in training_dynamics.items():
    if metric == "MP":
      simplicity[id] = np.mean([m["confidence"] for m in epoch_metrics])
    elif metric == "EC":  
      n_epochs = len(epoch_metrics)
      f = None # First correct epoch
      for i, m in enumerate(epoch_metrics):
        if m["correct"] == 1:
          f = i + 1
          break
      if f is not None:
        simplicity[id] = n_epochs / f
      else:
        simplicity[id] = 0.0

  return simplicity