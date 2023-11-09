import os, argparse

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
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--TASK', type=str, default="LiftCube-v0", \
        help="choose one task from: drawer-open-v2, drawer-close-v2, window-open-v2, window-close-v2, button-press-v2, sweep-into-v2, door-unlock-v2, door-close-v2, handle-pull-v2, handle-press-v2, handle-press-side-v2")
    parser.add_argument('--FILE_PATH', type=str, default=None)

    args = parser.parse_args()

    # File path to save result
    if args.FILE_PATH == None:
        args.FILE_PATH = "results/metaworld/{}.txt".format(args.TASK)
    os.makedirs(args.FILE_PATH, exist_ok=True)

    code_generator = ZeroShotGenerator(METAWORLD_PROMPT)
    general_code, specific_code = code_generator.generate_code(instruction_mapping[args.TASK], mapping_dicts)

    with open(os.path.join(args.FILE_PATH, "general.py"), "w") as f:
        f.write(general_code)

    with open(os.path.join(args.FILE_PATH, "specific.py"), "w") as f:
        f.write(specific_code)