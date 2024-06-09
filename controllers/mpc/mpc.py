import os
import yaml
import numpy as np
import scipy
import torch
import datetime

from robots.robot import QuadrupedRobot
from controllers.low_level.foot_planner import FootPlanner


class MPC:
	def __init__(self, env, use_gpu):
		self.env = env

		# Device information
		self.device = torch.device("cuda" if use_gpu else "cpu")

		# State information
		self.num_states = self.env.observation_space['policy'].shape[1]
		self.num_actions = env.action_space.shape[1]

		# Robot information
		self.robot = QuadrupedRobot("a1")
		self.num_legs = 4
		self.num_joints_per_leg = 3
		self.g = 9.81

		# The state space is 12 (base position, base orientation, base velocity, base angular velocity)
		self.mpc_state = np.zeros((12,))

		# Construct the full path to the config file
		current_dir = os.path.dirname(os.path.realpath(__file__))
		with open(os.path.join(current_dir, "mpc_config.yaml")) as file:
			self.config = yaml.load(file, Loader=yaml.FullLoader)

		# Extract MPC and Gait parameters
		self.horizon = self.config['mpc']["horizon"]
		self.dt = self.config['mpc']["dt"]
		self.gait_period = int(self.config['gait']["period"] / self.dt)
		self.duty_cycles = self.config['gait']["duty_cycles"]
		self.phase_offsets = self.config['gait']["phase_offsets"]

		# Initialize the foot planner
		self.foot_planner = FootPlanner(self.robot, self.num_legs, self.horizon, self.dt, self.gait_period, self.duty_cycles, self.phase_offsets)

		# Initialize the contact schedule
		self.nominal_contact_schedule = self.foot_planner.init_nominal_contact_schedule()

		# Initialize time and planning index
		self.initial_timestamp = datetime.datetime.now()
		self.current_plan_index = 0
		self.first_element_duration = self.dt

	def run(self):
		# Reset the environment
		states, _ = self.env.reset()

		# Prepare the MPC state
		# We only need the base position, base orientation, base velocity, and base angular velocity
		mpc_state, cmd_vel, joint_pos, joint_vel = self._prepare_mpc_state(states)

		# START THE MPC LOOP
		# Update the planning index
		self.initial_timestamp = datetime.datetime.now()
		self._update_planning_index()

		# Get the reference trajectory
		ref_body_plan = self._get_reference(mpc_state, cmd_vel)

		# Compute the contact schedule
		contact_schedule = self.foot_planner.compute_contact_schedule(self.current_plan_index, self.nominal_contact_schedule)

		# Compute the footholds
		footholds = self.foot_planner.compute_foot_plan(contact_schedule, ref_body_plan, simple=False)

		while True:
			self.env.step(torch.randn((1, self.num_actions))*10)

		self.env.close()

	@staticmethod
	def _prepare_mpc_state(states):
		"""
		Processes the states from the environment to get the MPC state
		"""
		mpc_state_raw = states['policy'][0]
		base_pos = mpc_state_raw[0:3].cpu().numpy()
		base_orn_quat = mpc_state_raw[3:7].cpu().numpy()
		base_lin_vel = mpc_state_raw[7:10].cpu().numpy()
		base_ang_vel = mpc_state_raw[10:13].cpu().numpy()

		# Convert the quaternion to euler angles
		rotation = scipy.spatial.transform.rotation.Rotation.from_quat(base_orn_quat)
		base_orn_rpy = rotation.as_euler('xyz', degrees=False)

		mpc_state = np.concatenate([base_pos, base_orn_rpy, base_lin_vel, base_ang_vel], axis=0)

		# Extract other states information
		# We only have a planar cmd_vel (x,y,yaw) , so we set all the other 3 components to zero
		cmd_vel = mpc_state_raw[13:16].cpu().numpy()
		cmd_vel = np.concatenate([cmd_vel[0:2], np.zeros(3), cmd_vel[2:3]], axis=0)
		joint_pos = mpc_state_raw[16:28].cpu().numpy()
		joint_vel = mpc_state_raw[28:40].cpu().numpy()

		return mpc_state, cmd_vel, joint_pos, joint_vel

	def _get_reference(self, state, cmd_vel):
		"""
		Get the reference trajectory for the MPC controller given a desired velocity
		:param state: The current state of the robot (12x1)
		:param cmd_vel: The desired velocity twist (6x1)
		"""
		# Extract the cmd_vel
		lin_vel_x = cmd_vel[0]
		lin_vel_y = cmd_vel[1]
		lin_vel_z = cmd_vel[2]
		ang_vel_x = cmd_vel[3]
		ang_vel_y = cmd_vel[4]
		ang_vel_z = cmd_vel[5]

		# Extract the state
		base_pos_x = state[0]
		base_pos_y = state[1]
		base_pos_z = state[2]
		base_orn_r = state[3]
		base_orn_p = state[4]
		base_orn_y = state[5]
		base_lin_vel_x = state[6]
		base_lin_vel_y = state[7]
		base_lin_vel_z = state[8]
		base_ang_vel_x = state[9]
		base_ang_vel_y = state[10]
		base_ang_vel_z = state[11]

		# 4x3 = 12 (number of states)
		ref_body_plan = np.zeros((self.horizon, 12))

		# Fill out the first row of the reference body plan
		# Position
		ref_body_plan[0, 0] = base_pos_x
		ref_body_plan[0, 1] = base_pos_y
		ref_body_plan[0, 2] = state[2]  # We don't have rough terrain, so we take the desired z to be the current state

		# Orientation (RPY)
		ref_body_plan[0, 3] = 0
		ref_body_plan[0, 4] = 0
		ref_body_plan[0, 5] = base_orn_y

		# Linear velocity (rotated to local frame using 2D rotation matrix for yaw)
		ref_body_plan[0, 6] = lin_vel_x * np.cos(base_orn_y) - lin_vel_y * np.sin(base_orn_y)
		ref_body_plan[0, 7] = lin_vel_x * np.sin(base_orn_y) + lin_vel_y * np.cos(base_orn_y)
		ref_body_plan[0, 8] = lin_vel_z

		# Angular velocity
		ref_body_plan[0, 9] = ang_vel_x
		ref_body_plan[0, 10] = ang_vel_y
		ref_body_plan[0, 11] = ang_vel_z

		# Fill out the rest of the reference body plan by forward integrating (Forward Euler)
		for i in range(1, self.horizon):
			current_cmd_vel = np.copy(cmd_vel)

			# Rotate the current_cmd_vel to the local frame with yaw
			yaw = ref_body_plan[i - 1, 5]
			current_cmd_vel[0] = cmd_vel[0] * np.cos(yaw) - cmd_vel[1] * np.sin(yaw)
			current_cmd_vel[1] = cmd_vel[0] * np.sin(yaw) + cmd_vel[1] * np.cos(yaw)

			for j in range(6):
				if i == 1:
					ref_body_plan[i, j] = ref_body_plan[i - 1, j] + current_cmd_vel[j] * self.first_element_duration
				else:
					ref_body_plan[i, j] = ref_body_plan[i - 1, j] + current_cmd_vel[j] * self.dt
				ref_body_plan[i, j + 6] = current_cmd_vel[j]

		return ref_body_plan

	def _update_planning_index(self):
		# Compute the duration from when we start to now
		duration = (datetime.datetime.now() - self.initial_timestamp).total_seconds()
		self.current_plan_index = int(duration / self.dt)
		self.first_element_duration = (self.current_plan_index + 1) * self.dt - duration


def run_mpc(env, use_gpu):
	"""
	Run the MPC controller on a gym environment
	"""
	# Initialize the MPC controller
	mpc = MPC(env, use_gpu)
	mpc.run()
