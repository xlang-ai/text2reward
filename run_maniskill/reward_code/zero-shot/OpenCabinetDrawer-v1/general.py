import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action):
    # Define weights for different parts of the reward
    w_distance = 0.4
    w_goal = 0.4
    w_action = 0.2

    # Calculate distance between robot's gripper and the cabinet handle
    handle_pcd = self.cabinet.handle.get_world_pcd()
    ee_cords = self.robot.get_ee_coords().reshape(-1, 3)
    distance = cdist(ee_cords, handle_pcd).min()
    distance_reward = -w_distance * distance  # Negative reward since we want to minimize the distance

    # Calculate the difference between current state of cabinet drawer and its goal state
    # Positive reward since we want to maximize the qpos
    goal_diff = self.cabinet.handle.qpos - self.cabinet.handle.target_qpos
    goal_reward = w_goal * goal_diff

    # Add regularization of robot's action, penalize large actions
    action_reward = -w_action * np.linalg.norm(action)

    # Check if the target drawer is static, if so, give a large positive reward
    if self.cabinet.handle.check_static():
        static_reward = 1.0
    else:
        static_reward = 0.0

    # Combine different parts of the reward
    reward = distance_reward + goal_reward + action_reward + static_reward

    return reward