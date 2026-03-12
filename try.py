from experiments.xil_loop import exp_xil_loop
from functions.xil import random_sampling, simplicity_sampling

SEED = 123
MODEL = "LeNet"
MOD_VARIANT = "modern"
DATASET = "DecoyFashionMNIST"
VARIANT = 2
SAMPLING_STRAT = simplicity_sampling
BUDGET = 30000
STEP = 100
INITIAL_QUERY = 0
LOG_NAME = f"ss_{MODEL}_{MOD_VARIANT}_{DATASET}"


exp_xil_loop(
  seed=SEED,
  model_name=MODEL,
  model_variant=MOD_VARIANT,
  dataset=DATASET,
  variation=VARIANT,
  sampling_strategy=SAMPLING_STRAT,
  budget=BUDGET,
  step=STEP,
  initial_query=INITIAL_QUERY,
  log_filename=LOG_NAME)