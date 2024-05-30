import gymnasium as gym
from isaac_ppo import PPO, Hyperparameters

from omni.isaac.orbit.app.app_launcher import AppLauncher
from utils.argparser import get_argparser

"""
Launch Isaac Sim as global variables
"""
parser = get_argparser()

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# Parse the arguments
args_cli = parser.parse_args()

# Launch Omniverse
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from envs.velocity.velocity_env_cfg import ActionsEffortCfg

# Import the necessary modules after Isaac Sim is launched
from omni.isaac.orbit_tasks.utils import parse_env_cfg


def main():
	# Parse the arguments
	# Environment configuration
	env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)

	# Modify the action space to be our joint effort instead of joint position
	# env_cfg.actions = ActionsEffortCfg()

	# Create the environment
	env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

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
	agent.learn(max_steps=1000)
	# agent.simulate('/home/dyuman/Downloads/model_100.pt')


if __name__ == '__main__':
	main()
