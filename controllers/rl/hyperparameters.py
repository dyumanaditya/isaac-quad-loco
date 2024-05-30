from isaac_ppo import Hyperparameters


def get_flat_hyperparameters():
	hyperparameters = Hyperparameters()
	hyperparameters.actor_hidden_sizes = [128, 128, 128]
	hyperparameters.critic_hidden_sizes = [128, 128, 128]
	return hyperparameters


def get_rough_hyperparameters():
	hyperparameters = Hyperparameters()
	hyperparameters.actor_hidden_sizes = [512, 256, 128]
	hyperparameters.critic_hidden_sizes = [512, 256, 128]
	return hyperparameters
