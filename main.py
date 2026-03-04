import torch
from model.lenet import load_lenet
from dataset.dataset import load_data, create_dataloaders
from functions.optimizer import load_optimizer
from functions.loss import load_loss_fun
from functions.functions import train_model, eval_model, save_checkpoint
from functions.xil import reset_model
from utils.utils import enable_reproducibility

# For separability
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

def plot_output(model, train_loader):
  model.eval()
  outputs_list = []
  labels_list = []
  is_confounded_list = []

  with torch.no_grad():
    for indeces, imgs, targets, masks in train_loader:
      imgs = imgs.to(device)
      outputs = model(imgs, return_features=True)

      outputs_list.append(outputs.cpu().numpy())
      labels_list.append(targets.numpy())

      mask_sum = masks.view(masks.size(0), -1).sum(dim=1)
      is_confounded = (mask_sum > 0).numpy()
      is_confounded_list.append(is_confounded)
  
  outputs_all = np.concatenate(outputs_list, axis=0)
  labels_all = np.concatenate(labels_list, axis=0)
  is_confounded_all = np.concatenate(is_confounded_list, axis=0)

  unique_classes = np.unique(labels_all)

  for c in unique_classes:
    class_mask = (labels_all == c)
    class_outputs = outputs_all[class_mask]
    class_confounded = is_confounded_all[class_mask]
             
    pca = PCA(n_components=2)
    reduced_outputs = pca.fit_transform(class_outputs)
        
    plt.figure(figsize=(8, 6))
        
    plt.scatter(
      reduced_outputs[~class_confounded, 0], 
      reduced_outputs[~class_confounded, 1], 
      c='green', label='Unconfounded (Minority)', alpha=0.8, marker='o'
    )
        
    plt.scatter(
      reduced_outputs[class_confounded, 0], 
      reduced_outputs[class_confounded, 1], 
      c='red', label='Confounded (Majority)', alpha=0.3, marker='x'
    )
        
    plt.title(f'PCA of Model Outputs for Class {c}')
    plt.xlabel('1st Principal Component')
    plt.ylabel('2nd Principal Component')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(f"plot/pca_class_{c}.pdf")

    #plt.show()



# Experiment seed
SEED = 123

DATASET_NAMES = ["DecoyMNIST", "DecoyFashionMNIST"]

# Params
DATASET = DATASET_NAMES[1]

	
if __name__ == "__main__":
  enable_reproducibility(SEED)
  use_cuda = torch.cuda.is_available()
  device = 'cuda' if use_cuda else 'cpu'

  model = load_lenet(device)
  save_checkpoint("reset_model", model)
  reset_model(model, device)

  train_data, val_data, test_data = load_data(
    DATASET, 
    seed=SEED, 
    reload=True, 
    bias_ratio=[0.995]*10
  )

  params = {"batch_size":32}
  m_params = [params]*3
  train_loader, val_loader, test_loader = create_dataloaders([train_data, val_data, test_data], m_params)

  # Train to get dynamics
  optim = load_optimizer("SGD", model.parameters(), lr=1e-3, weight_decay=0)
  ce_loss = load_loss_fun("CrossEntropy")
  log, dyn = train_model(model, train_loader, optim, ce_loss, 2, val_loader, device=device)

  loss, acc = eval_model(model, test_loader, ce_loss, device)
  print(f"Initial performance: {acc:.2f}")

  # Inspect outputs to see separablity in confounded and unconfounded samples
  plot_output(model, train_loader)

  




  
  
  