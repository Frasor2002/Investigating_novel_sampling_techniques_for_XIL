from experiments.confounder_study import exp_confounder_study

if __name__ == "__main__":
  SEED = 123
  MODEL="ModernLeNet"
  DATASET="DecoyMNIST"
  BIAS_RATIO=[0,0.99,0,0,0,0,0,0,0.99,0.99] #[0.99]*10 #[0,0.99,0,0,0,0.99,0,0,0.99,0.99]
  CONF_TYPES=[2] # [0,1,2]

  for conf in CONF_TYPES:
    exp_confounder_study(
      seed=SEED,
      model_name=MODEL,
      dataset=DATASET,
      bias_ratio=BIAS_RATIO,
      conf_type=conf,
      add="1"
    )
    exp_confounder_study(
      seed=SEED,
      model_name=MODEL,
      dataset=DATASET,
      bias_ratio=[0,0.5,0,0,0,0,0,0,0.5,0.5],
      conf_type=conf,
      add="2"
    )
    exp_confounder_study(
      seed=SEED,
      model_name=MODEL,
      dataset=DATASET,
      bias_ratio=[0,0.99,0.99,0.99,0.99,0.99,0,0,0.99,0.99],
      conf_type=conf,
      add="3"
    )
    
