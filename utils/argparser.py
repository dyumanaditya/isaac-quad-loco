"""
Function to gather CLI arguments
"""
import argparse


def get_argparser():
	parser = argparse.ArgumentParser(description="Quadruped Locomotion training using RL and Isaac Sim Orbit")
	parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to run in parallel")
	parser.add_argument("--task", type=str, default=None, help="Task to run the RL training on")
	parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
	parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
	parser.add_argument("--video_length", type=int, default=1000, help="Length of the recorded video (in steps).")
	parser.add_argument("--video_interval", type=int, default=5000, help="Interval between video recordings (in steps).")

	return parser
