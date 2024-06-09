from isaac_ppo import PPO
from utils.utils import most_recently_modified_directory, most_recently_modified_policy
from controllers.rl.hyperparameters import get_rough_hyperparameters


def play_rl(env, args_cli):
	# Create the hyperparameters object
	hyperparameters = get_rough_hyperparameters()

	# Create the agent
	device = 'gpu' if not args_cli.cpu else 'cpu'
	agent = PPO(env, hyperparameters,
				log_dir='logs', device=device,
				record_video=args_cli.video, video_length=args_cli.video_length, video_save_freq=args_cli.video_interval)

	# Simulate
	# Find the most recently modified policy file
	if args_cli.policy_file is None:
		most_recent_log = most_recently_modified_directory('./logs')
		policy_file = most_recently_modified_policy(most_recent_log)
	else:
		policy_file = args_cli.policy_file

	agent.simulate(policy_file)
