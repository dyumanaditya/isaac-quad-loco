from isaac_ppo import PPO, Hyperparameters


def train_rl(env, args_cli):
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

	# Learn
	agent.learn(max_steps=args_cli.max_steps)
