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

# Import the necessary modules after Isaac Sim is launched
from omni.isaac.orbit_tasks.utils import parse_env_cfg


def main():
	# Parse the arguments
	# Environment configuration
	env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)

	# Create the environment
	env = gym.make(args_cli.task, cfg=env_cfg)

	# Create the hyperparameters object
	hyperparameters = Hyperparameters()

	# Create the agent
	device = 'gpu' if not args_cli.cpu else 'cpu'
	agent = PPO(env, hyperparameters, log_dir='logs', device=device)

	# Learn
	agent.learn(max_steps=2000)
	# agent.simulate('policies/ppo_actor_critic_19660800.pth')


if __name__ == '__main__':
	main()
