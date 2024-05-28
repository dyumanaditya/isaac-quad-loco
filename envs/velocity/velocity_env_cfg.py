"""
This file re defines the environment configurations for the velocity environment given in
https://github.com/NVIDIA-Omniverse/orbit/blob/main/source/extensions/omni.isaac.orbit_tasks/omni/isaac/orbit_tasks/locomotion/velocity/velocity_env_cfg.py
For velocity tracking quadruped locomotion in Isaac Sim Orbit.

We provide modifications for the action space to be joint effort instead of joint position.
"""
from omni.isaac.orbit.utils import configclass
import omni.isaac.orbit_tasks.locomotion.velocity.mdp as mdp


@configclass
class ActionsEffortCfg:
	"""
	Action specifications for the MDP. We make this joint efforts instead of joint positions.
	"""
	joint_torque = mdp.JointEffortActionCfg(asset_name="robot", joint_names=[".*"], scale=5.0)
