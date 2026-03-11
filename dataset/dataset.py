from torch.utils.data.dataloader import DataLoader
from typing import List
from dataset.decoy_mnist import load_decoyMNIST
from dataset.decoy_fmnist import load_decoyFashionMNIST
from typing import Any
import matplotlib.pyplot as plt
import numpy as np

def visualize_k_samples(dataset: Any, label: int, k: int = 5) -> None:
  """Visualization helper to see k samples with a given label.
  Args:
    dataset (Any): dataset to visualize.
    label (int): label of which samples are shown.
    k (int): number of images to show. Defaults to 5.
  """
  fig, axes = plt.subplots(1, k, figsize=(4 * k, 4))
  
  if k == 1:
    axes = [axes]

  tr_indices = np.where(dataset.y == label)[0]  
  k_actual = min(k, len(tr_indices))
  selected_indices = np.random.permutation(tr_indices)[:k_actual]

  for a, i in enumerate(selected_indices):
    _, x, y, _ = dataset[i]
    
    axes[a].imshow(x.reshape(28, 28), cmap='gray')
    axes[a].set_xticks([])
    axes[a].set_yticks([])
    
    for spine in axes[a].spines.values():
      spine.set_edgecolor('gray')
      spine.set_linewidth(0.5)

  for a in range(k_actual, k):
    axes[a].axis('off')

  plt.tight_layout()



def load_data(name: str, **kwargs: Any):
  """Load a dataset.
  Args:
    name (str): name of the dataset.
    kwargs (Any): additional dataset arguments.
  Returns:
    tuple: train, val and test datasets.
  """
  datasets = {
    "DecoyMNIST": load_decoyMNIST,
    "DecoyFashionMNIST": load_decoyFashionMNIST
  }
  
  if name not in datasets.keys():
    raise ValueError("Wrong dataset name.")
  
  loader = datasets[name]
  train, val, test = loader(**kwargs)
  return train, val, test


def create_dataloaders(data_list: list, params_list: List[dict]) -> tuple:
  """Create dataloaders given train val and test and a list of params for each one.
  Args:
    data_list (list): list of datasets.
    params_list (list):list of dataloader params for each dataset.
  Returns:
    tuple: dataloaders.
  """
  loader_list = []
  for data, params in zip(data_list, params_list):
    data = data
    loader_list.append(DataLoader(data, **params))
  
  return tuple(loader_list)