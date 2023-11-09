import os, argparse

from langchain.prompts import PromptTemplate
from code_generation.single_flow.zero_shot.generation import ZeroShotGenerator
from code_generation.single_flow.classlike_prompt.MobileDualArmPrompt import MOBILE_DUAL_ARM_PROMPT
from code_generation.single_flow.classlike_prompt.MobilePandaPrompt import MOBILE_PANDA_PROMPT
from code_generation.single_flow.classlike_prompt.PandaPrompt import PANDA_PROMPT

franka_list = ["LiftCube-v0", "PickCube-v0", "StackCube-v0", "TurnFaucet-v0"]
mobile_list = ["OpenCabinetDoor-v1", "OpenCabinetDrawer-v1", "PushChair-v1"]

task_list = franka_list + mobile_list

LiftCube_Env = """
    self.cubeA : RigidObject # cube A in the environment
    self.cubeB : RigidObject # cube B in the environment
    self.cube_half_size = 0.02  # in meters
    self.robot : PandaRobot # a Franka Panda robot
    self.goal_height = 0.2 # in meters, indicate the z-axis height of our target
""".strip()

PickCube_Env = """
    self.cubeA : RigidObject # cube A in the environment
    self.cubeB : RigidObject # cube B in the environment
    self.cube_half_size = 0.02  # in meters
    self.robot : PandaRobot # a Franka Panda robot
    self.goal_position : np.ndarray[(3,)] # indicate the 3D position of our target position
""".strip()

StackCube_Env = """
    self.cubeA : RigidObject # cube A in the environment
    self.cubeB : RigidObject # cube B in the environment
    self.cube_half_size = 0.02  # in meters
    self.robot : PandaRobot # a Franka Panda robot
""".strip()

TurnFaucet_Env = """
    self.faucet : ArticulateObject # faucet in the environment
    self.faucet.handle : LinkObject # the handle of the faucet in the environment
    self.robot : PandaRobot # a Franka Panda robot
""".strip()


prompt_mapping = {
    "LiftCube-v0": PromptTemplate(input_variables=["instruction"], template=PANDA_PROMPT.replace("<environment_description>", LiftCube_Env)),
    "PickCube-v0": PromptTemplate(input_variables=["instruction"], template=PANDA_PROMPT.replace("<environment_description>", PickCube_Env)),
    "StackCube-v0": PromptTemplate(input_variables=["instruction"], template=PANDA_PROMPT.replace("<environment_description>", StackCube_Env)),
    "TurnFaucet-v0": PromptTemplate(input_variables=["instruction"], template=PANDA_PROMPT.replace("<environment_description>", TurnFaucet_Env)),
    "OpenCabinetDoor-v1": MOBILE_PANDA_PROMPT,
    "OpenCabinetDrawer-v1": MOBILE_PANDA_PROMPT,
    "PushChair-v1": MOBILE_DUAL_ARM_PROMPT,
}

instruction_mapping = {
    "LiftCube-v0": "Pick up cube A and lift it up by 0.2 meter.",
    "PickCube-v0": "Pick up cube A and move it to the 3D goal position.",
    "StackCube-v0": "Pick up cube A and place it on cube B. The task is finished when cube A is on top of cube B stably (i.e. cube A is static) and isn’t grasped by the gripper.",
    "TurnFaucet-v0": "Turn on a faucet by rotating its handle. The task is finished when qpos of faucet handle is larger than target qpos.",
    "OpenCabinetDoor-v1": "A single-arm mobile robot needs to open a cabinet door. The task is finished when qpos of cabinet door is larger than target qpos.",
    "OpenCabinetDrawer-v1": "A single-arm mobile robot needs to open a cabinet drawer. The task is finished when qpos of cabinet drawer is larger than target qpos.",
    "PushChair-v1": "A dual-arm mobile robot needs to push a swivel chair to a target location on the ground and prevent it from falling over.",
}

mapping_dicts_mapping = {
    "LiftCube-v0": {
        "self.cubeA.check_static()": "check_actor_static(self.obj)",
        "self.cubeA" : "self.obj",
        "self.robot.ee_pose" : "self.tcp.pose",
        "self.robot.check_grasp" : "self.agent.check_grasp",
        "self.cube_half_size" : "0.02",
        "self.robot.qpos" : "self.agent.robot.get_qpos()[:-2]",
        "self.robot.qvel" : "self.agent.robot.get_qvel()[:-2]",
    },
    "PickCube-v0": {
        "self.cubeA.check_static()": "check_actor_static(self.obj)",
        "self.cubeA" : "self.obj",
        "self.robot.ee_pose" : "self.tcp.pose",
        "self.robot.check_grasp" : "self.agent.check_grasp",
        "self.goal_position" : "self.goal_pos",
        "self.cube_half_size" : "0.02",
        "self.robot.qpos" : "self.agent.robot.get_qpos()[:-2]",
        "self.robot.qvel" : "self.agent.robot.get_qvel()[:-2]",
    },
    "StackCube-v0": {
        "self.robot.ee_pose": "self.tcp.pose",
        "self.robot.check_grasp": "self.agent.check_grasp",
        "self.goal_position": "self.goal_pos",
        "self.cube_half_size": "0.02",
        "self.robot.qpos": "self.agent.robot.get_qpos()[:-2]",
        "self.robot.qvel": "self.agent.robot.get_qvel()[:-2]",
        "self.robot.gripper_openness": "self.agent.robot.get_qpos()[-1] / self.agent.robot.get_qlimits()[-1, 1]",
        "self.cubeA.check_static()": "check_actor_static(self.cubeA)",
        "self.cubeB.check_static()": "check_actor_static(self.cubeB)",
    },
    "TurnFaucet-v0": {
        "self.faucet.handle.target_qpos": "self.target_angle",
        "self.faucet.handle.qpos": "self.current_angle",
        "self.faucet.handle.get_world_pcd()": "transform_points(self.target_link.pose.to_transformation_matrix(), self.target_link_pcd)",
        "self.robot.lfinger.get_world_pcd()": "transform_points(self.lfinger.pose.to_transformation_matrix(), self.lfinger_pcd)",
        "self.robot.rfinger.get_world_pcd()": "transform_points(self.rfinger.pose.to_transformation_matrix(), self.rfinger_pcd)",
        "self.faucet.handle.get_local_pcd()": "self.target_link_pcd",
        "self.robot.lfinger.get_local_pcd()": "self.lfinger_pcd",
        "self.robot.rfinger.get_local_pcd()": "self.rfinger_pcd",
        "self.robot.lfinger": "self.lfinger",
        "self.robot.rfinger": "self.rfinger",
        "self.faucet.handle": "self.target_link",
        "self.robot.ee_pose": "self.tcp.pose",
        "self.robot.qpos": "self.agent.robot.get_qpos()[:-2]",
        "self.robot.qvel": "self.agent.robot.get_qvel()[:-2]",
        "self.robot.check_grasp": "self.agent.check_grasp",
        "self.robot.gripper_openness": "self.agent.robot.get_qpos()[-1] / self.agent.robot.get_qlimits()[-1, 1]",
    },
    "OpenCabinetDoor-v1": {
        "self.robot.ee_pose": "self.agent.hand.pose",
        "self.robot.base_position": "self.agent.base_pose.p[:2]",
        "self.robot.base_velocity": "self.agent.base_link.velocity[:2]",
        "self.robot.qpos": "self.agent.robot.get_qpos()[:-2]",
        "self.robot.qvel": "self.agent.robot.get_qvel()[:-2]",
        "self.robot.get_ee_coords()": "self.agent.get_ee_coords()",
        "self.robot.gripper_openness": "self.agent.robot.get_qpos()[-1] / self.agent.robot.get_qlimits()[-1, 1]",
        "self.cabinet.handle.get_world_pcd()": "transform_points(self.target_link.pose.to_transformation_matrix(), self.target_handle_pcd)",
        "self.cabinet.handle.get_local_pcd()": "self.target_handle_pcd",
        "self.cabinet.handle.local_sdf": "self.target_handle_sdf.signed_distance",
        "self.cabinet.handle.target_grasp_poses": "self.target_handles_grasp_poses[self.target_link_idx]",
        "self.cabinet.handle.qpos": "self.link_qpos",
        "self.cabinet.handle.qvel": "self.link_qvel",
        "self.cabinet.handle.target_qpos": "self.target_qpos",
        "self.cabinet.handle.check_static()": "self.check_actor_static(self.target_link, max_v=0.1, max_ang_v=1)",
        "self.cabinet.handle": "self.target_link",
        # "self.cabinet": "TODO",
    },
    "OpenCabinetDrawer-v1": {
        "self.robot.ee_pose": "self.agent.hand.pose",
        "self.robot.base_position": "self.agent.base_pose.p[:2]",
        "self.robot.base_velocity": "self.agent.base_link.velocity[:2]",
        "self.robot.qpos": "self.agent.robot.get_qpos()[:-2]",
        "self.robot.qvel": "self.agent.robot.get_qvel()[:-2]",
        "self.robot.get_ee_coords()": "self.agent.get_ee_coords()",
        "self.robot.gripper_openness": "self.agent.robot.get_qpos()[-1] / self.agent.robot.get_qlimits()[-1, 1]",
        "self.cabinet.handle.get_world_pcd()": "transform_points(self.target_link.pose.to_transformation_matrix(), self.target_handle_pcd)",
        "self.cabinet.handle.get_local_pcd()": "self.target_handle_pcd",
        "self.cabinet.handle.local_sdf": "self.target_handle_sdf.signed_distance",
        "self.cabinet.handle.target_grasp_poses": "self.target_handles_grasp_poses[self.target_link_idx]",
        "self.cabinet.handle.qpos": "self.link_qpos",
        "self.cabinet.handle.qvel": "self.link_qvel",
        "self.cabinet.handle.target_qpos": "self.target_qpos",
        "self.cabinet.handle.check_static()": "self.check_actor_static(self.target_link, max_v=0.1, max_ang_v=1)",
        "self.cabinet.handle": "self.target_link",
        # "self.cabinet": "TODO",
    },
    "PushChair-v1": {
        "self.robot.get_ee_coords()": "self.agent.get_ee_coords()",
        "self.chair.get_pcd()": "self.env.env._get_chair_pcd()",
        "self.chair.check_static()": "self.check_actor_static(self.root_link, max_v=0.1, max_ang_v=0.2)",
        "self.chair": "self.root_link",
    },
}

if __name__ == '__main__':
    # add and parse argument
    parser = argparse.ArgumentParser()

    parser.add_argument('--TASK', type=str, default="LiftCube-v0", \
        help="choose one task from: LiftCube-v0, PickCube-v0, TurnFaucet-v0, OpenCabinetDoor-v1, OpenCabinetDrawer-v1, PushChair-v1")
    parser.add_argument('--FILE_PATH', type=str, default=None)

    args = parser.parse_args()

    # File path to save result
    if args.FILE_PATH == None:
        args.FILE_PATH = "results/maniskill-zeroshot/{}.txt".format(args.TASK)

    os.makedirs(args.FILE_PATH, exist_ok=True)

    code_generator = ZeroShotGenerator(prompt_mapping[args.TASK])
    general_code, specific_code = code_generator.generate_code(instruction_mapping[args.TASK], mapping_dicts_mapping[args.TASK])

    with open(os.path.join(args.FILE_PATH, "general.py"), "w") as f:
        f.write(general_code)

    with open(os.path.join(args.FILE_PATH, "specific.py"), "w") as f:
        f.write(specific_code)
