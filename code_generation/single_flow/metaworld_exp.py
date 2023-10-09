import os

from code_generation.single_flow.zero_shot.generation import ZeroShotGenerator
from code_generation.single_flow.classlike_prompt.MetaworldPrompt import METAWORLD_PROMPT


instruction_mapping = {
    "window-open-v2" : "Push and open a sliding window by its handle.",
    "window-close-v2" : "Push and close a sliding window by its handle.",
    "door-close-v2" : "Close a door with a revolving joint by pushing door's handle.",
    "drawer-open-v2" : "Open a drawer by its handle.",
    "drawer-close-v2" : "Close a drawer by its handle.",
    "door-unlock-v2" : "Unlock the door by rotating the lock counter-clockwise.",
    "sweep-into-v2" : " Sweep a puck from the initial position into a hole.",
    "button-press-v2" : "Press a button in y coordination.",
    "handle-press-v2" : "Press a handle down.",
    "handle-press-side-v2" : "Press a handle down sideways.",
}

mapping_dicts = {
    "self.robot.ee_position": "obs[:3]",
    "self.robot.gripper_openness": "obs[3]",
    "self.obj1.position": "obs[4:7]",
    "self.obj1.quaternion": "obs[7:11]",
    "self.obj2.position": "obs[11:14]",
    "self.obj2.quaternion": "obs[14:18]",
    "self.goal_position": "self.env._get_pos_goal()",
}


if __name__ == '__main__':
    # Task to run
    TASK_SET = [
    ]

    for TASK in TASK_SET:
        # File path to save result
        FILE_PATH = "results/metaworld/{}.txt".format(TASK)
        os.makedirs(FILE_PATH, exist_ok=True)

        code_generator = ZeroShotGenerator(METAWORLD_PROMPT)
        general_code, specific_code = code_generator.generate_code(instruction_mapping[TASK], mapping_dicts)

        with open(os.path.join(FILE_PATH, "general.py"), "w") as f:
            f.write(general_code)

        with open(os.path.join(FILE_PATH, "specific.py"), "w") as f:
            f.write(specific_code)