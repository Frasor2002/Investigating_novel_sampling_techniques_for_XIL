from experiments.training_dynamic import exp_train_dynamics
from experiments.model_output import exp_model_outputs
from experiments.explaination_entropy import exp_explaination_entropy
from experiments.utils import log_results, plot_results

# Experiment seed
SEED = 123
SEEDS = [123,111,222,333,444]
MODEL_NAMES = ["LeNet", "MLP"]
MODEL_NAME = MODEL_NAMES[1]
MODEL_VARIANTS = ["classic", "modern"]
MODEL_VARIANT = MODEL_VARIANTS[0]
DATASET_NAMES = ["DecoyMNIST", "DecoyFashionMNIST"]
DATASET = DATASET_NAMES[0]
DATASET_VARIANTS = [0,1,2]
VARIATION = DATASET_VARIANTS[2]

def run(seed,model,model_variant, dataset, variation):
  res1 = exp_train_dynamics(seed=seed, model_name=model,model_variant=model_variant, dataset=dataset, variation=variation)
  log_results(res1, f"td_{seed}_{model}{model_variant}_{dataset}_{variation}")

  #res2 = exp_model_outputs(seed=seed, model_name=model,model_variant=model_variant, dataset=dataset, variation=variation)
  #log_results(res2, f"mo_{seed}_{model}{model_variant}_{dataset}_{variation}")

  #res3 = exp_explaination_entropy(seed=seed, model_name=model,model_variant=model_variant, dataset=dataset, variation=variation)
  #log_results(res3, f"ee_{seed}_{model}{model_variant}_{dataset}_{variation}")
	
if __name__ == "__main__":
  #run(123, "LeNet", "modern", dataset="DecoyMNIST", variation=2)

  for seed in SEEDS:
    for data in DATASET_NAMES:
      for data_var in DATASET_VARIANTS:
        for model in MODEL_NAMES:
          if model == "LeNet":
            for model_var in MODEL_VARIANTS:
              run(seed, model, model_var, data, data_var)
          else:
            run(seed, model, "", data, data_var)
  
