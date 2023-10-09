import json
import os
import argparse
from code_generation.single_flow.zero_shot_exp import prompt_mapping, instruction_mapping, mapping_dicts_mapping

from code_generation.interactive.basic.generation import BasicHumanFeedbackCodeGenerator

if __name__ == '__main__':
    # add and parse argument
    parser = argparse.ArgumentParser()

    parser.add_argument('--TASK', type=str, default="StackCube-v0")
    parser.add_argument('--FILE_PATH', type=str, default="results/human_feedback_basic/StackCube-v0")

    args = parser.parse_args()

    # File path to save result
    os.makedirs(args.FILE_PATH, exist_ok=True)

    # Load the feedback history
    with open(os.path.join("data", "feedback_basic", args.TASK, "trajectory.jsonl"), "r") as f:
        history = [json.loads(line) for line in f.readlines()]

    code_generator = BasicHumanFeedbackCodeGenerator(prompt_mapping[args.TASK])
    general_code, specific_code = code_generator.generate_code(instruction_mapping[args.TASK],
                                                               mapping_dicts_mapping[args.TASK],
                                                               history)

    with open(os.path.join(args.FILE_PATH, "general.py"), "w") as f:
        f.write(general_code)

    with open(os.path.join(args.FILE_PATH, "specific.py"), "w") as f:
        f.write(specific_code)
