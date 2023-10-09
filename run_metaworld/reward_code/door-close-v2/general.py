import numpy as np

def compute_dense_reward(self, action, obs) -> float:
    # Calculate the distance between the end-effector and the door handle
    distance_to_handle = np.linalg.norm(self.robot.ee_position - self.obj1.position)

    # Calculate the distance between the door handle's current position and the goal position
    distance_to_goal = np.linalg.norm(self.obj1.position - self.goal_position)

    # Reward for reaching the door handle
    reach_reward = -distance_to_handle

    # Reward for pushing the door handle towards the goal position
    push_reward = -distance_to_goal

    # Encourage the gripper to close when near the door handle
    gripper_reward = 0
    if distance_to_handle < 0.1:
        gripper_reward = -self.robot.gripper_openness

    # Combine the rewards
    reward = reach_reward + push_reward + gripper_reward
    return reward