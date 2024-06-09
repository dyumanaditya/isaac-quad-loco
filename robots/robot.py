import numpy as np
import scipy
import rbdl

from controllers.low_level.robot_driver import RobotDriver


class QuadrupedRobot:
    def __init__(self, name):
        self.name = name
        self.num_legs = 4
        self.num_joints_per_leg = 3
        self.toe_radius = 0.02

        self.robot_driver = RobotDriver(name, self.num_legs, self.num_joints_per_leg)
        self.robot_model = self.robot_driver.robot_model

        # Initialize lists to store geometry data
        self.body_id_list = []
        self.leg_idx_list = []
        self.legbase_offsets = []
        self.l0_vec = []
        self.g_body_legbases = []
        self.body_name_list = ["toe0", "toe1", "toe2", "toe3"]
        self.hip_name_list = ["hip0", "hip1", "hip2", "hip3"]
        self.upper_name_list = ["upper0", "upper1", "upper2", "upper3"]
        self.lower_name_list = ["lower0", "lower1", "lower2", "lower3"]
        self.toe_name_list = ["toe0", "toe1", "toe2", "toe3"]

        self._compute_transforms()

        # Store some of the initialized data to robot driver
        self.robot_driver.legbase_offsets = self.legbase_offsets
        self.robot_driver.body_id_list = self.body_id_list
        self.robot_driver.leg_idx_list = self.leg_idx_list

    def _compute_transforms(self):
        for leg_idx in range(self.num_legs):
            # Get the body ID for the toe
            self.body_id_list.append(self.robot_model.GetBodyId(self.body_name_list[leg_idx]))

            # From body COM to abad
            # self.legbase_offsets.append(self.robot.get_transform(self.upper_name_list[leg_idx])[:, 3][:3])
            self.legbase_offsets.append(
                self.robot_model.X_base[self.robot_model.GetBodyId(self.hip_name_list[leg_idx])].r)

            # From abad (lateral movement) to hip (forward backward movement) (Y coordinate only)
            # self.l0_vec.append(self.robot.get_transform(self.upper_name_list[leg_idx])[:, 3][1])
            self.l0_vec.append(self.robot_model.X_base[self.robot_model.GetBodyId(self.upper_name_list[leg_idx])].r[1])

        # From hip to knee
        # hip_to_knee = self.robot.get_transform(self.lower_name_list[leg_idx])[:, 3][:3]
        # upper_link_length = np.max(np.abs(hip_to_knee))
        # knee_offset = hip_to_knee
        # hip_to_knee = self.robot_model.X_base[self.robot_model.GetBodyId(self.lower_name_list[leg_idx])].r
        # upper_link_length = np.max(np.abs(hip_to_knee))
        # knee_offset = hip_to_knee

        # From knee to toe
        # knee_to_toe = self.robot.get_transform(self.toe_name_list[leg_idx])[:, 3][:3]
        # lower_link_length = np.max(np.abs(knee_to_toe))
        # foot_offset = knee_to_toe
        # knee_to_toe = self.robot_model.X_base[self.robot_model.GetBodyId(self.toe_name_list[leg_idx])].r
        # lower_link_length = np.max(np.abs(knee_to_toe))
        # foot_offset = knee_to_toe

        # Sort the leg indices based on the body IDs
        self.leg_idx_list = list(range(4))
        self.leg_idx_list.sort(key=lambda i: self.body_id_list[i])

        # Compute transforms from body to leg bases
        for leg_offset in self.legbase_offsets:
            translation = np.array(leg_offset)
            rotation = scipy.spatial.transform.Rotation.from_euler('z', 0).as_matrix()
            transform = np.eye(4)
            transform[:3, :3] = rotation
            transform[:3, 3] = translation
            self.g_body_legbases.append(transform)

    def get_world_hip_position(self, leg_idx, body_pos, body_rpy):
        """
        Calculates the position of a leg's hip joint in the world frame.
        :param leg_idx: Index of the leg
        :param body_pos: Position of the body in the world frame
        :param body_rpy: Roll, pitch, yaw of the body in the world frame
        :return: Position of the hip joint in the world frame
        """
        # Create transformation matrix world to body
        rotation = scipy.spatial.transform.Rotation.from_euler('xyz', body_rpy).as_matrix()
        world_body_transform = np.eye(4)
        world_body_transform[:3, :3] = rotation
        world_body_transform[:3, 3] = body_pos

        # Compute transform from body to legbase but offset (y-coord) by l0
        body_legbase_transform = self.g_body_legbases[leg_idx]
        body_legbase_transform[1, 3] += self.l0_vec[leg_idx]

        # Compute transform for offset leg base relative to the world frame
        world_hip_transform = world_body_transform @ body_legbase_transform

        # Return the position of the hip joint in the world frame
        return world_hip_transform[:3, 3]
