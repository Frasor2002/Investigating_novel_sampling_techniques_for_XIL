import torch
import torch.nn as nn
import numpy as np
from captum.attr import Saliency, InputXGradient, IntegratedGradients, Attribution
from captum.attr import visualization as viz
from typing import Optional, Any
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

def visualize_k_expl(all_attr: torch.Tensor, all_imgs: torch.Tensor, dataset: Any, target_label:int, k: int=3):
  """Visualizer helper to plot 5 attributions given a label.
  Args:
    all_attr (Tensor): all attributions.
    all_imgs (Tensor): all imgs to plot.
    dataset (Any): dataset to plot.
    target_label (int): label of which to plot explainations.
    k (int): number of explainations to show.
  """

  class_indices = np.where(dataset.y == target_label)[0]
  if len(class_indices) < k:
    print(f"Not enough samples for class {target_label}")
    selected_indices = class_indices
  else:
    selected_indices = np.random.permutation(class_indices)[:k]
  fig, axes = plt.subplots(1, k, figsize=(20, 5))
  for plot_i, data_idx in enumerate(selected_indices):
    visualize_explanation(
      attr=all_attr[data_idx], 
      img=all_imgs[data_idx],
      title=f"Class: {target_label}",
      plt_fig_axis=(fig, axes[plot_i]), 
      use_pyplot=False
    )
  plt.tight_layout()


def get_method(name: str, model: nn.Module) -> Attribution:
  """Get attribution method.
  Args:
    name (str): method name.
    model (nn.Module): model to explain.
  Returns:
    Attribution: attribution method.
  """
  attr_methods = {
    'input gradient': Saliency,
    'input X gradient': InputXGradient,
    'integrated gradient': IntegratedGradients,
  }

  if name not in attr_methods.keys():
    raise ValueError("Wrong explanation method name.")
  
  method_class = attr_methods[name]
  method = method_class(model)
  
  return method


def compute_explanation(method_name: str, model: nn.Module, inputs: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
  """Compute the explaination given a method name.
  Args:
    method_name (str): explaination method name.
    model (Module): model to explain.
    inputs (Tensor): batch of inputs to explain.
    targets (Optional[Tensor]): optional labels, if not present use model predictions.
  Returns:
    Tensor: computed explainations.
  """
  # Get model current device
  model.eval()
  device = next(model.parameters()).device

  # Expect an input of shape Batch_size X Channel X Height X Width
  inputs = inputs.to(device)
  explainer = get_method(method_name, model)

  # If no target is provided get predictions for inputs and use those
  if targets is None:
    logits = model(inputs)
    targets = logits.argmax(dim=1)

  attributions = explainer.attribute(inputs, targets)
  return attributions


def visualize_explanation(attr: torch.Tensor, img: torch.Tensor, **kwargs: Any) -> None:
  """Given a attributions and images, visualize the explaination.
  Args:
    attr (Tensor): either a single or a batch of attributions.
    img (Tensor): either a single or a batch of imgs.
  """

  def format_for_viz(tensor: torch.Tensor) -> np.ndarray:
    """Format a tensor for visualization."""
    if tensor.dim() == 4:
      tensor = tensor.squeeze(0) # Remove batch dimension
    # Permute to dimensions (H, W, C)
    return tensor.permute(1, 2, 0).detach().to("cpu").numpy()
  
  # Convert to correct format
  attr_np = format_for_viz(attr)
  img_np = format_for_viz(img)

  #print(f"Visualizing | Image Shape: {img_np.shape} | Attr Shape: {attr_np.shape}")

  # Call Captum Visualization
  viz.visualize_image_attr(
    attr_np,
    original_image=img_np,
    method="heat_map",
    show_colorbar=True,
    sign="absolute_value",
    outlier_perc=5,
    **kwargs
  )


def explain_dataset(loader: DataLoader, model: nn.Module, device: str="cpu") -> tuple:
  """Compute explaination for an entire dataset for visualization.
  Args:
    loader (DataLoader): dataloader with data to explain.
    model (Module): model to use for explaiantion.
    device (str): device where to compute explainations.
  """
  model.eval()
  attr_lists = []
  imgs_lists = []

  loop = tqdm(loader, desc="Explaining", leave=False)
  for indices, imgs, targets, masks in loop:
    imgs = imgs.to(device)
    imgs.requires_grad_(True)
    attrs = compute_explanation("input gradient", model, imgs) 
    imgs.requires_grad_(False)

    # Save attrs
    attr_lists.append(attrs.detach().cpu())
    all_attr = torch.cat(attr_lists, dim=0)

    # Save imgs
    imgs_lists.append(imgs.detach().cpu())
    all_images = torch.cat(imgs_lists, dim=0)

  return all_attr, all_images


def evaluate_explainations(pred_expl: torch.Tensor, gt_expl: torch.Tensor) -> Any:
  """Evaluate model explaination by computing a penalty.
  Args:
    pred_expl (Tensor): model attributions.
    gt_expl (Tensor): masks that contain confounder location.
  Returns:
    Any: penalty score.
  """
  pred = pred_expl.squeeze().detach().cpu().numpy()
  gt = gt_expl.squeeze().detach().cpu().numpy()

  pred_abs = np.abs(pred)

  attribution_on_confounder = np.sum(pred_abs * gt)
  total_attribution = np.sum(pred_abs)

  epsilon = 1e-8
  penalty = attribution_on_confounder / (total_attribution + epsilon)

  return float(penalty)

  
