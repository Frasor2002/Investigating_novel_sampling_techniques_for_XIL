from experiments.training_dynamic import exp_train_dynamics

	
if __name__ == "__main__":
  SEED = 123
  MODEL="ModernLeNet"
  DATASET="DecoyMNIST"
  BIAS_RATIO=[0,0.99,0.99,0.99,0.99,0.99,0,0,0.99,0.99] #[0.99]*10

  exp_train_dynamics(
    seed=SEED,
    model_name=MODEL,
    dataset=DATASET,
    bias_ratio=BIAS_RATIO,
    conf_type=2,
    metric="MP"
  )
  
