import yourdfpy
import numpy as np
import scipy
import pinocchio as pin
import rbdl


# Load the URDF model
robot = yourdfpy.URDF.load("robots/a1_description/urdf/a1.urdf")
rbdl_model = rbdl.loadModel("robots/a1_description/urdf/a1.urdf")
body_name_list = ["toe0", "toe1", "toe2", "toe3"]

body_id_list = []
for i in range(4):
    body_id_list.append(rbdl_model.GetBodyId(body_name_list[i]))

leg_idx_list = list(range(4))
leg_idx_list.sort(key=lambda i: body_id_list[i])
# print(leg_idx_list)



# pin_robot = pin.buildModelFromUrdf("robots/a1_description/urdf/a1.urdf")
# pin_data = pin_robot.createData()
#
# nq = pin_robot.nq
# nv = pin_robot.nv
# print(f"Number of DOFs: nq={nq}, nv={nv}")
#
# # Compute forward kinematics
# q = pin.neutral(pin_robot)
# # pin.forwardKinematics(pin_robot, pin_data, q)
# # pin.updateFramePlacements(pin_robot, pin_data)
#
# # Get the frame ID for the 'toe0' frame
# frame_name = 'toe0'
# if frame_name in [frame.name for frame in pin_robot.frames]:
#     frame_id = pin_robot.getFrameId(frame_name)
#     print(frame_id)
# else:
#     raise ValueError(f"Frame '{frame_name}' not found in the model.")
#
# jacobian_frame = pin.computeFrameJacobian(pin_robot, pin_data, q, frame_id)
# print(jacobian_frame)
# print()
#
# # Compute the Jacobian of the specified frame
# jacobian = np.array(pin.computeJointJacobians(pin_robot, pin_data, q))
# # jacobian = pin.computeJointJacobian(pin_robot, pin_data, q, frame_id)
# print(jacobian)
# print(jacobian == jacobian_frame)
# print()
# # pin.getJointJacobian()
#
# # print(pin_robot.getFrameId('toe0'))
# # frame_id = pin_robot.getFrameId('jtoe0')
# # pin.computeJointJacobian(pin_robot, pin_data, q, frame_id)
# # print(pin.getJointJacobian(pin_robot, pin_data, frame_id, pin.LOCAL))

# Initialize lists to store geometry data
legbase_offsets = []
l0_vec = []
hip_name_list = ["hip0", "hip1", "hip2", "hip3"]
upper_name_list = ["upper0", "upper1", "upper2", "upper3"]
lower_name_list = ["lower0", "lower1", "lower2", "lower3"]
toe_name_list = ["toe0", "toe1", "toe2", "toe3"]

q_size = rbdl_model.q_size
q = np.zeros(q_size)
qd = np.zeros(q_size)
qdd = np.zeros(q_size)
tau = np.zeros(q_size)
# data = rbdl_model.createData()
rbdl.ForwardDynamics(rbdl_model, q, qd, tau, qdd)


# Loop through the legs
for leg_idx in range(4):
    # From body COM to abad
    body_id = rbdl_model.GetBodyId(hip_name_list[leg_idx])
    tform = rbdl_model.X_base[body_id]
    legbase_offsets.append(tform.r)
    # print(tform.r)
    # print()
    # print(robot.get_transform(hip_name_list[leg_idx])[:, 3][:3])

    # tform = rbdl_model.X_base[rbdl_model.GetBodyId(upper_name_list[leg_idx])]
    # print(tform.r)
    # print(robot.get_transform(upper_name_list[leg_idx])[:, 3][:3])

    # From abad (lateral movement) to hip (forward backward movement) (Y coordinate only)
    l0_vec.append(rbdl_model.X_base[rbdl_model.GetBodyId(upper_name_list[leg_idx])].r[1])
    # print(rbdl_model.X_base[rbdl_model.GetBodyId(upper_name_list[leg_idx])].r[1])
    # print(robot.get_transform(upper_name_list[leg_idx])[:, 3][1])
    # print()

    # From hip to knee
    hip_to_knee = rbdl_model.X_base[rbdl_model.GetBodyId(lower_name_list[leg_idx])].r
    upper_link_length = np.max(np.abs(hip_to_knee))
    knee_offset = hip_to_knee

    # From knee to toe
    # print(rbdl_model.X_base[rbdl_model.GetBodyId(toe_name_list[leg_idx])])
    # knee_to_toe = rbdl_model.X_base[rbdl_model.GetBodyId(toe_name_list[leg_idx])].r
    # lower_link_length = np.max(np.abs(knee_to_toe))
    # foot_offset = knee_to_toe





    # # From body COM to hip
    # legbase_offsets.append(robot.get_transform(upper_name_list[leg_idx])[:, 3][:3])
    #
    # # From abad (lateral movement) to hip (forward backward movement) (Y coordinate only)
    # l0_vec.append(robot.get_transform(upper_name_list[leg_idx])[:, 3][1])
    #
    # # From hip to knee
    # hip_to_knee = robot.get_transform(lower_name_list[leg_idx])[:, 3][:3]
    # upper_link_length = np.max(np.abs(hip_to_knee))
    # knee_offset = hip_to_knee
    #
    # # From knee to toe
    # knee_to_toe = robot.get_transform(toe_name_list[leg_idx])[:, 3][:3]
    # lower_link_length = np.max(np.abs(knee_to_toe))
    # foot_offset = knee_to_toe
    #
    # # print(robot.get_transform(upper_name_list[leg_idx]))
    # # print()
    # # print(pin_robot.getFrameId(upper_name_list[leg_idx]))
    # # print(pin_data.[pin_robot.getFrameId(upper_name_list[leg_idx])])

# Abad offset from legbase
abad_offset = np.array([0, 0, 0])

# Compute transforms
g_body_legbases = []
for leg_offset in legbase_offsets:
    translation = np.array(leg_offset)
    rotation = scipy.spatial.transform.Rotation.from_euler('z', 0).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    g_body_legbases.append(transform)

jac_block = np.zeros((3, 18))
q = np.random.rand(19)
rbdl.CalcPointJacobian(rbdl_model, q, body_id_list[0], np.zeros(3), jac_block)
# print(jac_block)
