import torch
from model.lenet import load_lenet
from dataset.dataset import load_data, create_dataloaders
from functions.optimizer import load_optimizer
from functions.loss import load_loss_fun
from functions.functions import train_model, eval_model, save_checkpoint
from functions.xil import xil_loop, random_sampling, simplicity_sampling, reset_model
from utils.utils import enable_reproducibility

# Experiment seed
SEED = 123

DATASET_NAMES = ["DecoyMNIST", "DecoyFashionMNIST"]
SAMPLING_STRATS = [random_sampling, simplicity_sampling]

# Params
DATASET = DATASET_NAMES[0]
RR_REG = 1 if DATASET == "DecoyMNIST" else 1e-3
STRATEGY = SAMPLING_STRATS[1]
START_QUERY = 0
STEP_SIZE = 1
BUDGET = 2000

	
if __name__ == "__main__":
  enable_reproducibility(SEED)
  use_cuda = torch.cuda.is_available()
  device = 'cuda' if use_cuda else 'cpu'

  # Run XIL loop
  model = load_lenet(device)
  save_checkpoint("reset_model", model)
  reset_model(model, device)

  train_data, val_data, test_data = load_data(
    DATASET, 
    seed=SEED, 
    reload=True, 
    bias_ratio=[0.95]*10
  )

  params = {"batch_size":32}
  m_params = [params]*3
  train_loader, val_loader, test_loader = create_dataloaders([train_data, val_data, test_data], m_params)

  # Train to get dynamics
  optim = load_optimizer("SGD", model.parameters(), lr=1.0e-2, weight_decay=0)
  ce_loss = load_loss_fun("CrossEntropy")
  log, dyn = train_model(model, train_loader, optim, ce_loss, 10, val_loader, device=device)

  loss, acc = eval_model(model, test_loader, ce_loss, device)
  print(f"Initial performance: {acc:.2f}")

  query = xil_loop(
    train_data=train_data,
    model=model, 
    sampling_strategy=STRATEGY,
    budget=BUDGET,
    val_loader=val_loader,
    test_loader=test_loader,
    tr_dynamics=dyn,
    step_size=STEP_SIZE,
    starting_query=START_QUERY,
    rrr_reg_rate=RR_REG,
    device=device
  )






  
  
  