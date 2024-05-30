"""
Function to gather CLI arguments
"""
import argparse


def get_argparser():
	parser = argparse.ArgumentParser(description="Quadruped Locomotion training using RL and Isaac Sim Orbit")
	parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to run in parallel")
	parser.add_argument("--task", type=str, default=None, help="Task to run the RL training on")
	parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
	parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
	parser.add_argument("--video_length", type=int, default=1000, help="Length of the recorded video (in steps).")
	parser.add_argument("--video_interval", type=int, default=1000, help="Interval between video recordings (in steps).")
	parser.add_argument("--mode", type=str, default="rl", help="Mode to run the script in. Options: rl, mpc, rl-mpc")
	parser.add_argument("--play_mode", action="store_true", default=False, help="Play mode to visualize a trained policy.")
	parser.add_argument("--max_steps", type=int, default=None, help="Maximum number of steps to run the training for. (only applied for rl or rl+mpc mode)")
	parser.add_argument("--policy_file", type=str, default=None, help="Policy file to load for playing.")

	return parser
