from scipy.stats import pearsonr
import numpy as np
from numpy.typing import ArrayLike
import os
import matplotlib.pyplot as plt

def compute_correlations(separation_list: ArrayLike, is_confounded: ArrayLike, labels: ArrayLike) -> dict:
  """Function to compute correlation between a separation strategy and actual confounder
  presence.
  Args:
    separation_list (ArrayLike): list that tells results of the separation method. 
    is_confounded (ArrayLike): list with gt results of confounded and not-confounded.
    labels (ArrayLike): list with the labels for each sample for class-wise correlation.
  Returns:
    dict: total and classwise correlation. 
  """

  separation_list = np.array(separation_list)
  is_confounded = np.array(is_confounded)
  labels = np.array(labels)
  
  total_corr = pearsonr(separation_list, is_confounded)

  # Class-wise correlation
  class_corr = {}
  unique_classes = np.unique(labels)

  for label in unique_classes:
    class_mask = (labels == label)
    c_scores = separation_list[class_mask]
    c_conf = is_confounded[class_mask]
        
    class_corr[int(label)] = pearsonr(c_scores, c_conf)

  return {
    "total": total_corr,
    "class": class_corr
  }



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "log")
PLOT_DIR = os.path.join(LOG_DIR, "plot")

def log_results(result: dict, filename: str) -> None:
  path = os.path.join(LOG_DIR, f"{filename}.log")
  os.makedirs(LOG_DIR, exist_ok=True)

  with open(path, 'w', encoding='utf-8') as f:
    total_corr = result['total']
    class_corr = result['class']
    
    f.write(f"Total correlation: stat={total_corr[0]:.4f} | pval={total_corr[1]}")
    f.write("\n\n")
    for key, val in class_corr.items():
      f.write(f"Class correlation for label {key}: stat={val[0]:.4f} | pval={val[1]}")
      f.write("\n")
      

# TODO utility to plot the different corralation for visualization
def plot_results(result: dict, filename: str) -> None:
  pass
  
