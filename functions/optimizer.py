from torch.optim import SGD, Adam, AdamW, Optimizer
from typing import Iterable, Any

def load_optimizer(name: str, params: Iterable, **kwargs: Any) -> Optimizer:
  """Load optimizer.
  Args:
    name (str): optimizer name.
    params (Iterable): model params to optimize.
    kwargs** (Any): additional optimizer arguments.
  Returns:
    Optimizer: optimizer initialized.
  """

  optimizers = {
    "SGD": SGD,
    "Adam": Adam,
    "AdamW": AdamW
  }

  if name not in optimizers.keys():
    raise ValueError("Wrong optimizer name.")
  
  optim_class = optimizers[name]
  optim = optim_class(params, **kwargs)

  return optim
