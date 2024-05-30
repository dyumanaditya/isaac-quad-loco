from isaac_ppo import PPO, Hyperparameters
from utils.utils import most_recently_modified_directory, most_recently_modified_policy


def play_rl(env, args_cli, policy_file_path=None):
	# Create the hyperparameters object
	hyperparameters = Hyperparameters()
	hyperparameters.actor_hidden_sizes = [128, 128, 128]
	hyperparameters.critic_hidden_sizes = [128, 128, 128]
	hyperparameters.num_transitions_per_env = 24

	# Create the agent
	device = 'gpu' if not args_cli.cpu else 'cpu'
	agent = PPO(env, hyperparameters,
				log_dir='logs', device=device,
				record_video=args_cli.video, video_length=args_cli.video_length, video_save_freq=args_cli.video_interval)

	# Simulate
	# Find the most recently modified policy file
	if policy_file_path is None:
		most_recent_log = most_recently_modified_directory('./logs')
		policy_file = most_recently_modified_policy(most_recent_log)
	else:
		policy_file = policy_file_path

	agent.simulate(policy_file)
