import gymnasium as gym

from omni.isaac.orbit.app.app_launcher import AppLauncher
from utils.argparser import get_argparser

# Import controllers
# RL
from controllers.rl.flat.train import train_rl as train_rl_flat
from controllers.rl.flat.play import play_rl as play_rl_flat
from controllers.rl.rough.train import train_rl as train_rl_rough
from controllers.rl.rough.play import play_rl as play_rl_rough

# MPC
from controllers.mpc.mpc import run_mpc

# RL+MPC

"""
START Launch Isaac Sim as global variables
"""
parser = get_argparser()

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# Parse the arguments
args_cli = parser.parse_args()

# Launch Omniverse
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from envs.mpc.velocity.velocity_flat_env_cfg import MPCActionsEffortCfg, MPCObservationsCfg

# Import the necessary modules after Isaac Sim is launched
from omni.isaac.orbit_tasks.utils import parse_env_cfg

"""
END Launch Isaac Sim as global variables
"""


def main():
	# Parse the arguments
	# Environment configuration
	env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)

	# Modify the environment configuration for mpc and rl-mpc
	if args_cli.mode == "mpc":
		env_cfg.actions = MPCActionsEffortCfg()
		env_cfg.observations = MPCObservationsCfg()
		env_cfg.scene.num_envs = 1
	elif args_cli.mode == "rl-mpc":
		pass

	# Create the environment
	env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

	# Decide which controller to use.
	# Use the RL controller
	if args_cli.mode == "rl":
		if 'Rough' in args_cli.task:
			if args_cli.play_mode:
				play_rl_rough(env, args_cli)
			else:
				train_rl_rough(env, args_cli)
		else:
			if args_cli.play_mode:
				play_rl_flat(env, args_cli)
			else:
				train_rl_flat(env, args_cli)

	# Use the MPC controller
	elif args_cli.mode == "mpc":
		run_mpc(env, use_gpu=not args_cli.cpu)

	# Use the hybrid RL+MPC controller
	elif args_cli.mode == "rl-mpc":
		pass


if __name__ == '__main__':
	main()
