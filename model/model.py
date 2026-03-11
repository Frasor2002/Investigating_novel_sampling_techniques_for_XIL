from model.resnet import load_resnet
from model.lenet import load_lenet
from model.mlp import load_mlp
from typing import Any
from torch.nn import Module

def load_model(name: str, **kwargs: Any) -> Module:
  """Load a model"""

  model_loaders = {
    "LeNet": load_lenet,
    "MLP": load_mlp,
    "ResNet": load_resnet
  }

  if name not in model_loaders.keys():
    raise ValueError("Wrong model name.")
  
  loader = model_loaders[name]
  model = loader(**kwargs)

  return model

