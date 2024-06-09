import numpy as np
import rbdl
import scipy


class RobotDriver:
    def __init__(self, name, num_legs, num_joints_per_leg):
        self.name = name
        self.num_legs = num_legs
        self.num_joints_per_leg = num_joints_per_leg
        self.robot_model = None

        # Robot and transform data
        self.legbase_offsets = []
        self.body_id_list = []
        self.leg_idx_list = []

        if self.name == 'a1':
            self.robot_model = rbdl.loadModel("robots/a1_description/urdf/a1.urdf")
        else:
            raise ValueError(f"Invalid robot name: {self.name}")

        # Initialize RBDL frames by doing a forward kinematics pass
        q_size = self.robot_model.q_size
        q = np.zeros(q_size)
        qd = np.zeros(q_size)
        qdd = np.zeros(q_size)
        tau = np.zeros(q_size)
        rbdl.ForwardDynamics(self.robot_model, q, qd, tau, qdd)

    def compute_inverse_dynamics(self, joint_angles, joint_velocities, body_pos, body_quat, body_lin_vel, body_ang_vel,
                                 foot_accelerations, grfs, contact_mode):
        """
        Computes the inverse dynamics for the robot given the joint angles, velocities, accelerations, and body state.
        :param joint_angles: Joint angles (12,)
        :param joint_velocities: Joint velocities (12,)
        :param body_pos: Body position (3,)
        :param body_quat: Body orientation in quaternion form (4,)
        :param body_lin_vel: Body linear velocity (3,)
        :param body_ang_vel: Body angular velocity (3,)
        :param foot_accelerations: Foot accelerations (12,)
        :param grfs: Ground reaction forces (12,)
        :param contact_mode: Whether the leg is in contact with the ground (4,)
        :return: Joint torques
        """
        # Initialize states
        # 0-11: Joint angles, 12-14: Body position, 15-18: Body quaternion
        state_positions = np.concatenate([joint_angles, body_pos, body_quat])
        state_velocities = np.concatenate([joint_velocities, body_lin_vel, body_ang_vel])

        # Convert into RBDL format
        # One extra dimension for q because of quaternion representation
        q = np.zeros(19)
        q_dot = np.zeros(18)

        # Body Position, Body Linear Velocity
        q[:3] = state_positions[12:15]
        q_dot[:3] = state_velocities[12:15]

        # Body Quaternion (only 1st 3 elements, the last one comes at the end)
        q[3:6] = state_positions[15:18]
        q[18] = state_positions[18]

        # Body Angular Velocity
        q_dot[3:6] = state_velocities[15:18]

        # Joint angles and velocities for each leg
        for leg_idx in range(self.num_legs):
            q[6 + leg_idx * 3: 9 + leg_idx * 3] = state_positions[leg_idx * 3: (leg_idx + 1) * 3]
            q_dot[6 + leg_idx * 3: 9 + leg_idx * 3] = state_velocities[leg_idx * 3: (leg_idx + 1) * 3]

        # Compute Jacobian that relates the joint velocities to the velocity of the foot
        # The Jacobian has dimensions: 12 x 18.
        # The Jacobian maps the joint velocities to the foot velocities. Or maps the joint space to the task space.
        # The joint space is 3x4 + 1x6 = 18 dimensions (6 for the floating base, 12 for the joints of the 4 legs)
        # The task space is 3x4=12 dimensions (3 dimensions for each leg).
        jacobian = np.zeros((12, 18))

        # Compute the Jacobian for each leg
        for leg_idx in range(self.num_legs):
            jac_block = np.zeros((3, 18))
            rbdl.CalcPointJacobian(self.robot_model, q, self.body_id_list[leg_idx], np.zeros(3), jac_block)
            jacobian[leg_idx * 3: (leg_idx + 1) * 3, :] = jac_block

        # Compute the equivalent force in generalized coordinates
        tau_stance = -jacobian.T @ grfs

        # Compute equation of motion EOM
        M = np.zeros((18, 18))
        N = np.zeros(18)
        rbdl.CompositeRigidBodyAlgorithm(self.robot_model, q, M)
        rbdl.NonlinearEffects(self.robot_model, q, q_dot, N)

        # Compute J_dot*q_dot -- we need this to compute the torques necessary to reach the foot accelerations
        foot_acc_J_dot = np.zeros(12)
        for leg_idx in range(self.num_legs):
            foot_acc_J_dot[leg_idx * 3: (leg_idx + 1) * 3] = rbdl.CalcPointAcceleration(self.robot_model, q, q_dot,
                                                                                        np.zeros(18),
                                                                                        self.body_id_list[leg_idx],
                                                                                        np.zeros(3))

        # Each leg in contact with the ground imposes three constraints (one for each spatial dimension: x, y, and z).
        # The total number of constraints is three times the number of legs in contact with the ground
        # A: A matrix that will store the Jacobian rows corresponding to the legs in contact with the ground.
        # A_dot_q_dot: A vector that will store the time derivative of the Jacobian multiplied by the generalized
        # velocity vector for the legs in contact with the ground.

        num_constraints = 3 * np.sum(contact_mode)
        A = np.zeros((num_constraints, 18))
        A_dot_q_dot = np.zeros(num_constraints)
        constraint_cnt = 0
        for leg_idx in range(self.num_legs):
            if contact_mode[leg_idx]:
                A[3 * constraint_cnt: 3 * constraint_cnt + 3, :] = jacobian[leg_idx * 3: (leg_idx + 1) * 3, :]
                A_dot_q_dot[3 * constraint_cnt: 3 * constraint_cnt + 3] = foot_acc_J_dot[leg_idx * 3: (leg_idx + 1) * 3]
                constraint_cnt += 1

        # Compute acceleration from J * q_ddot
        # Effective foot acceleration given the desired foot acceleration and the current state
        foot_acc_q_ddot = foot_accelerations - foot_acc_J_dot

        # Compute damped Jacobian inverse (using pseudo inverse for simplicity)
        jacobian_inv = np.linalg.pinv(jacobian[:, 6:18])

        # Construct block matrix for EOM solution
        # This will be used to solve for the joint torques
        blk_mat = np.zeros((18 + num_constraints, 18 + num_constraints))

        # Fill in the block matrix
        # The first two 6x6 blocks of M are due to the body, the rest is due to the legs
        blk_mat[:6, :6] = -M[:6, :6] + np.dot(M[:6, 6:18], np.dot(jacobian_inv, jacobian[:, :6]))
        blk_mat[6:18, :6] = -M[6:18, :6] + np.dot(M[6:18, 6:18], np.dot(jacobian_inv, jacobian[:, :6]))

        # Fill in diagonal blocks for the swing legs, we can therefore decouple the swing legs from the stance legs
        for i in range(4):
            if not contact_mode[self.leg_idx_list[i]]:
                blk_mat[3 * i + 6:3 * i + 9, 3 * i + 6:3 * i + 9] = np.eye(3)

        # Fill the Top-Right and Bottom-Left Blocks with constraint related terms
        blk_mat[:18, 18:] = -A.T
        blk_mat[18:, :6] = -A[:, :6] + np.dot(A[:, 6:18], np.dot(jacobian_inv, jacobian[:, :6]))

        # Perform inverse dynamics (blk_mat * blk_sol = blk_vec) using SVD
        # Represents the right-hand side (RHS) of the linear system, which combines the dynamics and constraints of the robot.
        blk_vec = np.zeros(18 + num_constraints)

        blk_vec[:6] = N[:6] + np.dot(M[:6, 6:18], np.dot(jacobian_inv, foot_acc_q_ddot))
        blk_vec[6:18] = N[6:18] + np.dot(M[6:18, 6:18], np.dot(jacobian_inv, foot_acc_q_ddot)) - tau_stance[6:18]
        blk_vec[18:] = A_dot_q_dot + np.dot(A[:, :12], np.dot(jacobian_inv, foot_acc_q_ddot))

        blk_sol = np.linalg.lstsq(blk_mat, blk_vec, rcond=None)[0]
        tau_swing = blk_sol[6:18]

        # Convert the order back
        tau = np.zeros(12)
        for i in range(4):
            if contact_mode[self.leg_idx_list[i]]:
                tau[3 * self.leg_idx_list[i]:3 * self.leg_idx_list[i] + 3] = tau_stance[6 + 3 * i:9 + 3 * i]
            else:
                tau[3 * self.leg_idx_list[i]:3 * self.leg_idx_list[i] + 3] = tau_swing[3 * i:3 * i + 3]

        # Check for NaN or Inf, make zero if any
        if np.isnan(tau).any() or np.isinf(tau).any():
            tau.fill(0)

        return tau
