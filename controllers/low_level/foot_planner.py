import numpy as np


class FootPlanner:
    def __init__(self, robot, num_legs, horizon, dt, gait_period, duty_cycles, phase_offsets):
        self.robot = robot
        self.num_legs = num_legs
        self.horizon = horizon
        self.dt = dt
        self.gait_period = gait_period
        self.duty_cycles = duty_cycles
        self.phase_offsets = phase_offsets
        self.g = 9.81

    def init_nominal_contact_schedule(self):
        """
        Compute the nominal contact schedule based on the gait parameters. This will be used as a template for the
        contact schedule by shifting it in time.
        """
        nominal_contact_schedule = np.zeros((self.gait_period, self.num_legs), dtype=bool)

        # The duty cycle of a leg is the fraction of the total period during which the leg is in contact with the ground,
        # The phase offset is the fraction of the total period by which the contact schedule of a leg is shifted
        # relative to the other legs.

        for i in range(self.gait_period):
            for leg_idx in range(self.num_legs):
                leg_phase_offset = self.phase_offsets[leg_idx]
                leg_duty_cycle = self.duty_cycles[leg_idx]

                start_contact_phase = self.gait_period * leg_phase_offset
                end_contact_phase = self.gait_period * (leg_phase_offset + leg_duty_cycle)

                # Check if the current time step i is within the contact phase that starts and ends within the same period
                # Check if the current time step i is within the contact phase that starts in the previous period and ends in the current period
                if start_contact_phase <= i < end_contact_phase or i < end_contact_phase - self.gait_period:
                    nominal_contact_schedule[i, leg_idx] = True
                else:
                    nominal_contact_schedule[i, leg_idx] = False

        return nominal_contact_schedule

    def compute_contact_schedule(self, current_plan_index, nominal_contact_schedule):
        """
        Compute the contact schedule by shifting the nominal contact schedule in time
        """
        contact_schedule = np.zeros((self.horizon, self.num_legs), dtype=bool)

        # Phase shift the nominal contact schedule
        phase = current_plan_index % self.gait_period

        for i in range(self.horizon):
            contact_schedule[i] = nominal_contact_schedule[(i + phase) % self.gait_period]

        return contact_schedule

    def compute_foot_plan(self, contact_schedule, ref_body_plan, simple=False):
        """
        Compute the foot plan based on the contact schedule.
        This function looks at the reference plan and computes the footholds based on the Raibert heuristic:

        Simple:
        p_des = p_ref + v_com * delta_t * 0.5

        Complex:
        p_des = p_ref + centrifugal + velocity_tracking

        Where:
        centrifugal = (h/g) * (v_touchdown_ref x omega_ref)
        velocity_tracking = sqrt(h/g) * (v_touchdown_ref - v_touchdown)
        h = body_height_touchdown

        - p_ref is the location on the ground beneath the hip.
        - v_com is the velocity of the COM projected onto the ground (xy plane).
        """

        # Initialize the footholds to have current foot positions
        foot_plan = np.zeros((self.horizon, self.num_legs, 3))
        body_pos = ref_body_plan[0, 0:3]
        body_rpy = ref_body_plan[0, 3:6]
        for leg_idx in range(self.num_legs):
            foot_plan[0, leg_idx] = self.robot.get_world_hip_position(leg_idx, body_pos, body_rpy)

        for i in range(1, self.horizon):
            for leg_idx in range(self.num_legs):
                # Check if the leg has just touched down
                if self._is_new_contact(contact_schedule, i, leg_idx):
                    # Extract state information
                    p_ref = ref_body_plan[i, 0:3]
                    orn_ref = ref_body_plan[i, 3:6]
                    v_touchdown_ref = ref_body_plan[i, 6:9]
                    omega_ref = ref_body_plan[i, 9:12]
                    body_height_touchdown = p_ref[2]

                    # Compute the hip position
                    hip_pos = self.robot.get_world_hip_position(leg_idx, p_ref, orn_ref)

                    # Compute the desired foothold
                    if simple:
                        foot_plan[i, leg_idx] = hip_pos + v_touchdown_ref * self.gait_period * 0.5
                    else:
                        centrifugal = (body_height_touchdown / self.g) * np.cross(v_touchdown_ref, omega_ref)
                        velocity_tracking = np.sqrt(body_height_touchdown / self.g) * v_touchdown_ref
                        foot_plan[i, leg_idx] = hip_pos + centrifugal + velocity_tracking

                    # Adjust the z-coordinate of the foothold to be the ground height and toe radius
                    foot_plan[i, leg_idx][2] = self.robot.toe_radius

                # If it's not a new contact, keep the foothold the same as the previous time step
                else:
                    foot_plan[i, leg_idx] = foot_plan[i - 1, leg_idx]

        # Compute foot plan for swing legs (not in contact)

        return foot_plan

    @staticmethod
    def _is_new_contact(contact_schedule, horizon_idx, leg_idx):
        """
        Check if the current leg has just touched down. This is done by checking if the leg was not in contact in the
        previous time step but is in contact in the current time step.
        """
        if horizon_idx == 0:
            return False
        else:
            return contact_schedule[horizon_idx, leg_idx] and not contact_schedule[horizon_idx - 1, leg_idx]
