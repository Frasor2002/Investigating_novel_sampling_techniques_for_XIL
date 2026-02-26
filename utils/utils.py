import torch
import numpy as np
import random

def enable_reproducibility(seed: int = 123) -> None:
  """Set seeds for Python, NumPy and PyTorch.
  Args:
    seed (int): seed for reproducibility
  """
  random.seed(seed)
  torch.manual_seed(seed)
  np.random.seed(seed)