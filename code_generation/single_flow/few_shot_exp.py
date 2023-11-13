import argparse
import os

from code_generation.single_flow.classlike_prompt import PANDA_PROMPT_FOR_FEW_SHOT, MOBILE_PANDA_PROMPT_FOR_FEW_SHOT, \
    MOBILE_DUAL_ARM_PROMPT_FOR_FEW_SHOT
from code_generation.single_flow.few_shot.generation import FewShotGenerator
from code_generation.single_flow.zero_shot_exp import instruction_mapping, mapping_dicts_mapping, LiftCube_Env, \
    PickCube_Env, StackCube_Env, TurnFaucet_Env

few_shot_prompt_mapping = {
    "LiftCube-v0": PANDA_PROMPT_FOR_FEW_SHOT.replace("<environment_description>", LiftCube_Env),
    "PickCube-v0": PANDA_PROMPT_FOR_FEW_SHOT.replace("<environment_description>", PickCube_Env),
    "StackCube-v0": PANDA_PROMPT_FOR_FEW_SHOT.replace("<environment_description>", StackCube_Env),
    "TurnFaucet-v0": PANDA_PROMPT_FOR_FEW_SHOT.replace("<environment_description>", TurnFaucet_Env),
    "OpenCabinetDoor-v1": MOBILE_PANDA_PROMPT_FOR_FEW_SHOT,
    "OpenCabinetDrawer-v1": MOBILE_PANDA_PROMPT_FOR_FEW_SHOT,
    "PushChair-v1": MOBILE_DUAL_ARM_PROMPT_FOR_FEW_SHOT
}

gold_reward_path_mapping = {
    "LiftCube-v0": "lift_cube",
    "PickCube-v0": "pick_cube",
    "StackCube-v0": "stack_cube",
    "TurnFaucet-v0": "turn_faucet",
    "OpenCabinetDoor-v1": "open_cabinet_door",
    "OpenCabinetDrawer-v1": "open_cabinet_drawer",
    "PushChair-v1": "push_chair",
}


def load_all_examples(current_task: str, verbose=True):
    examples = []

    for task in few_shot_prompt_mapping.keys():
        # Skip current task to avoid information leaking
        if task == current_task:
            continue
        with open(os.path.join("../gold_reward_rewrite", gold_reward_path_mapping[task] + ".py"), "r") as f:
            instruction = instruction_mapping[task]
            reward_code = f.read()
            examples.append({"instruction": instruction, "reward_code": reward_code})
            if verbose:
                print("Load task: {}".format(task))

    return examples


if __name__ == "__main__":
    # add and parse argument
    parser = argparse.ArgumentParser()

    parser.add_argument('--TASK', type=str, default="LiftCube-v0", \
                        help="choose one task from: LiftCube-v0, PickCube-v0, TurnFaucet-v0, OpenCabinetDoor-v1, OpenCabinetDrawer-v1, PushChair-v1")
    parser.add_argument('--FILE_PATH', type=str, default=None)
    parser.add_argument('--MODEL_NAME', type=str, default="gpt-4")

    args = parser.parse_args()

    # File path to save result
    if args.FILE_PATH == None:
        args.FILE_PATH = "results/{}/maniskill-fewshot/{}.txt".format(args.MODEL_NAME, args.TASK)

    os.makedirs(args.FILE_PATH, exist_ok=True)

    code_generator = FewShotGenerator(
        few_shot_prompt_mapping[args.TASK],
        args.MODEL_NAME,
        examples=load_all_examples(current_task=args.TASK),
        k_examples=1,
    )
    general_code, specific_code = code_generator.generate_code(instruction_mapping[args.TASK],
                                                               mapping_dicts_mapping[args.TASK], )

    with open(os.path.join(args.FILE_PATH, "general.py"), "w") as f:
        f.write(general_code)

    with open(os.path.join(args.FILE_PATH, "specific.py"), "w") as f:
        f.write(specific_code)
