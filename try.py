from experiments.xil_loop import exp_xil_loop
from functions.xil import random_sampling, simplicity_sampling

SEED = 123
MODEL = "LeNet"
MOD_VARIANT = "modern"
DATASET1 = "DecoyMNIST"
DATASET2 = "DecoyFashionMNIST"
VARIANT = 2
BUDGET = 300
STEP = 10
INITIAL_QUERY = 0

# MNIST
RR_REG_SIM = 1e-1
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
  rr_reg=RR_REG_SIM,
  log_filename="ss_mnist")

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
    rr_reg=1,
    log_filename="rs_mnist")

if False:
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