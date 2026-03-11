import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

class ModernLeNet(nn.Module):
  """Modern reintepretation of LeNet."""

  def __init__(self) -> None:
    "Initialize the model."
    super(ModernLeNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, 1)
    self.conv2 = nn.Conv2d(32, 64, 3, 1)
    self.dropout1 = nn.Dropout(0.25)
    self.dropout2 = nn.Dropout(0.5)
    self.fc1 = nn.Linear(9216, 128)
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x: torch.Tensor, return_features:bool=False) -> torch.Tensor:
    """Forward pass through the model."""
    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.relu(x)
    x = F.max_pool2d(x, 2)
    x = self.dropout1(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = F.relu(x)

    if return_features:
      return x

    x = self.dropout2(x)
    x = self.fc2(x)
    return x # F.log_softmax(x, dim=1)
  


class LeNet(torch.nn.Module):
  """Traditional LeNet architecture."""

  def __init__(self):
    super(LeNet, self).__init__()
    # 1 input image channel (black & white), 6 output channels, 5x5 square convolution
    self.conv1 = torch.nn.Conv2d(1, 6, 5)
    self.conv2 = torch.nn.Conv2d(6, 16, 3)
    self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)  # 400 features from convolutions
    self.fc2 = torch.nn.Linear(120, 84)
    self.fc3 = torch.nn.Linear(84, 10)

  def forward(self, x: torch.Tensor, return_features: bool = False):
    x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
    x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    x = torch.flatten(x, 1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))

    if return_features:
      return x

    x = self.fc3(x)
    return x


def load_lenet(variant: str= "classic", device: str = "cuda") -> Union[ModernLeNet,LeNet]:
  """Load LeNet model.
  Args:
    variant (str): can be 'classic' or 'modern' variant.
    device (str): device where model is loaded.
  Returns:
    Union[LeNet,ModernLeNet]: model created.
  
  """
  if variant not in ['classic', 'modern']:
    raise ValueError("Wrong LeNet variant name.")
  
  if variant == 'classic':
    model = LeNet()
  else:
    model = ModernLeNet()

  model = model.to(device)
  return model
