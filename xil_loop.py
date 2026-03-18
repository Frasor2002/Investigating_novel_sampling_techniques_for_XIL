from experiments.xil_loop import exp_xil_loop

SEED = 123
MODEL="ModernLeNet"
DATASET="DecoyMNIST"
BIAS_RATIO= [0,0.99,0,0,0,0.99,0,0,0.99,0.99]
CONF_TYPE=2
SAMPLING="simplicity"
BUDGET=5000
STEP=100
INITIAL=0
RR_REG=1e-1

if __name__ == "__main__":

  exp_xil_loop(
    seed=SEED,
    model_name=MODEL,
    dataset=DATASET,
    bias_ratio=BIAS_RATIO,
    conf_type=CONF_TYPE,
    sampling_strategy="simplicity",
    budget=BUDGET,
    step=STEP,
    initial_query=INITIAL,
    rr_reg=RR_REG
  )

  exp_xil_loop(
    seed=SEED,
    model_name=MODEL,
    dataset=DATASET,
    bias_ratio=BIAS_RATIO,
    conf_type=CONF_TYPE,
    sampling_strategy="random",
    budget=BUDGET,
    step=STEP,
    initial_query=INITIAL,
    rr_reg=RR_REG
  )
