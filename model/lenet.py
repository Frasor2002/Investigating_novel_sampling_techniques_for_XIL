import torch
import torch.nn as nn
import torch.nn.functional as F

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

  def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    x = self.dropout2(x)
    x = self.fc2(x)
    return x # F.log_softmax(x, dim=1)


def load_lenet(device: str = "cuda") -> ModernLeNet:
  model = ModernLeNet()
  model = model.to(device)
  return model
