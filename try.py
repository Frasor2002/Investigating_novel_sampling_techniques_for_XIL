from experiments.xil_loop import exp_xil_loop
from functions.xil import random_sampling, simplicity_sampling

SEED = 123
MODEL = "LeNet"
MOD_VARIANT = "modern"
DATASET1 = "DecoyMNIST"
DATASET2 = "DecoyFashionMNIST"
VARIANT = 2
BUDGET = 2500
STEP = 100
INITIAL_QUERY = 0

# MNIST
# Random sampling
exp_xil_loop(
  seed=SEED,
  model_name=MODEL,
  model_variant=MOD_VARIANT,
  dataset=DATASET1,
  variation=VARIANT,
  sampling_strategy=random_sampling,
  budget=BUDGET,
  step=STEP,
  initial_query=INITIAL_QUERY,
  log_filename="rs_mnist")

# Simplicity
exp_xil_loop(
  seed=SEED,
  model_name=MODEL,
  model_variant=MOD_VARIANT,
  dataset=DATASET1,
  variation=VARIANT,
  sampling_strategy=simplicity_sampling,
  budget=BUDGET,
  step=STEP,
  initial_query=INITIAL_QUERY,
  log_filename="ss_mnist")

# FMNIST
# Random sampling
exp_xil_loop(
  seed=SEED,
  model_name=MODEL,
  model_variant=MOD_VARIANT,
  dataset=DATASET2,
  variation=VARIANT,
  sampling_strategy=random_sampling,
  budget=BUDGET,
  step=STEP,
  initial_query=INITIAL_QUERY,
  log_filename="rs_fmnist")

# Simplicity
exp_xil_loop(
  seed=SEED,
  model_name=MODEL,
  model_variant=MOD_VARIANT,
  dataset=DATASET2,
  variation=VARIANT,
  sampling_strategy=simplicity_sampling,
  budget=BUDGET,
  step=STEP,
  initial_query=INITIAL_QUERY,
  log_filename="ss_fmnist")