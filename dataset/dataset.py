from torch.utils.data.dataloader import DataLoader
from typing import List
from dataset.decoy_mnist import load_decoyMNIST
from dataset.decoy_fmnist import load_decoyFashionMNIST
from typing import Any
import matplotlib.pyplot as plt
import numpy as np

def visualize_5_samples(dataset: Any, label: int) -> None:
  """Visualization helper to see 5 samples with a given label.
  Args:
    dataset (Any): dataset to visualize.
    label (int): label of which samples are shown.
  """
  _, axes = plt.subplots(1, 5, figsize=(5*5, 5*1))

  tr_indices = np.where(dataset.y == label)[0]
  for a, i in enumerate(np.random.permutation(tr_indices)[:5]):
    _, x, y, _ = dataset[i]
    axes[a].imshow(x.reshape(28, 28), cmap='gray')
    axes[a].set_xticks([])
    axes[a].set_yticks([])
    #axes[a].set_xlabel(f'label: {y}')
    

def load_data(name: str, **kwargs: Any):
  # to choose a dataset with a dict and load that data

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
  # utility to create a dataloader
  loader_list = []
  for data, params in zip(data_list, params_list):
    data = data
    loader_list.append(DataLoader(data, **params))
  
  return tuple(loader_list)