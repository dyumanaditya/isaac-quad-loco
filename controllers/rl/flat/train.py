from isaac_ppo import PPO
from controllers.rl.hyperparameters import get_flat_hyperparameters


def train_rl(env, args_cli):
	# Create the hyperparameters object
	hyperparameters = get_flat_hyperparameters()

	# Create the agent
	device = 'gpu' if not args_cli.cpu else 'cpu'
	agent = PPO(env, hyperparameters,
				log_dir='logs', device=device,
				record_video=args_cli.video, video_length=args_cli.video_length, video_save_freq=args_cli.video_interval)

	# Learn
	if args_cli.max_steps is None:
		max_steps = 500
	else:
		max_steps = args_cli.max_steps
	agent.learn(max_steps=max_steps)
