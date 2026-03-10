from experiments.training_dynamic import exp_train_dynamics
from experiments.model_output import exp_model_outputs
from experiments.explaination_entropy import exp_explaination_entropy
from experiments.utils import log_results, plot_results

# Experiment seed
SEED = 123
DATASET_NAMES = ["DecoyMNIST", "DecoyFashionMNIST"]
DATASET = DATASET_NAMES[0]
VARIATION = 2

def run(seed, dataset, variation):
  res1 = exp_train_dynamics(seed=SEED, dataset=DATASET, variation=VARIATION)
  log_results(res1, "train_dynamics")

  res2 = exp_model_outputs(seed=SEED,dataset="DecoyMNIST", variation=VARIATION)
  log_results(res2, "model_output")

  res3 = exp_explaination_entropy(seed=SEED, dataset="DecoyMNIST", variation=VARIATION)
  log_results(res3, "explaination_entropy")
	
if __name__ == "__main__":
  run(SEED, DATASET, VARIATION)
