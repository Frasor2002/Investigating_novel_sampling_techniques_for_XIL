import torch

class MLP(torch.nn.Module):
  """Simple MLP model for MNIST datasets."""

  def __init__(self):
    super(MLP, self).__init__()
    # Input flattened size is 1 * 28 * 28 = 784

    self.linear1 = torch.nn.Linear(784, 200)
    self.activation = torch.nn.ReLU()
    self.linear2 = torch.nn.Linear(200, 10)

  def forward(self, x: torch.Tensor, return_features: bool = False):
    x = torch.flatten(x, start_dim=1)
    x = self.linear1(x)
    x = self.activation(x)

    if return_features:
      return x

    x = self.linear2(x)
    return x

def load_mlp(device: str = "cuda"):
  """Load the mlp model."""
  model = MLP()
  model = model.to(device)
  return model

