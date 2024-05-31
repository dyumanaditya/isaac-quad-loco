"""
MPC CONFIG
This file re defines the environment configurations for the velocity environment given in
https://github.com/NVIDIA-Omniverse/orbit/blob/main/source/extensions/omni.isaac.orbit_tasks/omni/isaac/orbit_tasks/locomotion/velocity/velocity_env_cfg.py
For velocity tracking quadruped locomotion in Isaac Sim Orbit.

We provide modifications for the action space to be joint effort instead of joint position.
"""
from omni.isaac.orbit.utils import configclass
import omni.isaac.orbit_tasks.locomotion.velocity.mdp as mdp
from omni.isaac.orbit.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.orbit.managers import ObservationTermCfg as ObsTerm


@configclass
class MPCActionsEffortCfg:
	"""
	Action specifications for the MDP. We make this joint efforts instead of joint positions.
	"""
	joint_torque = mdp.JointEffortActionCfg(asset_name="robot", joint_names=[".*"], scale=5.0)


@configclass
class MPCObservationsCfg:
	"""
	Observation specifications for the MDP. NO NOISE
	We modify the observations so that we give the MPC what it needs to run:

	Current Full State
	1. Base position
	2. Base velocity
	3. Base orientation
	4. Base angular velocity

	Others
	5. The desired velocity twist
	6. Joint positions
	7. Joint velocities

	"""
	@configclass
	class PolicyCfg(ObsGroup):
		"""Observations for policy group."""

		# 1. Base position
		base_pos = ObsTerm(func=mdp.root_pos_w)

		# 2. Base velocity
		base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

		# 3. Base orientation in quaternion form
		base_quat = ObsTerm(func=mdp.root_quat_w)

		# 4. Base angular velocity
		base_ang_vel = ObsTerm(func=mdp.base_ang_vel)

		# 5. The desired velocity twist
		velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

		# 6. Joint positions
		joint_pos = ObsTerm(func=mdp.joint_pos_rel)

		# 7. Joint velocities
		joint_vel = ObsTerm(func=mdp.joint_vel_rel)

		def __post_init__(self):
			self.enable_corruption = True
			self.concatenate_terms = True

	# observation groups
	policy: PolicyCfg = PolicyCfg()
