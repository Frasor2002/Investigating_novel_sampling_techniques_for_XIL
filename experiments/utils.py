from scipy.stats import pearsonr
import numpy as np
from numpy.typing import ArrayLike

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

# TODO utility to log the results of the experiment
def log_results(result):
  pass

# TODO utility to plot the different corralation for visualization
def plot_results(result):
  pass