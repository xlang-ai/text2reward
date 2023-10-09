import numpy as np
from scipy.spatial import distance

def compute_dense_reward(self, action, obs) -> float:

    # Calculate the Euclidean distance between the end-effector and the puck
    gripper_obj_dist = np.linalg.norm(obs[:3] - obs[4:7])

    # Calculate the Euclidean distance between the puck and the goal hole
    obj_goal_dist = np.linalg.norm(obs[4:7] - self.env._get_pos_goal())

    # The reward for getting the puck into the hole
    if obj_goal_dist < 0.05:  # Threshold for 'close enough'
        reward = 1.0
    else:
        # Reward is higher the closer the gripper is to the puck and the puck is to the goal
        # We want to minimize these distances, so we negate them
        # We also scale the distances by some factor to control their impact on the total reward
        reward = -0.01 * gripper_obj_dist - 0.01 * obj_goal_dist

    # Regularization of the robot's action
    # We want to encourage the robot to take smoother actions, so we penalize large actions
    action_penalty = 0.001 * np.sum(np.square(action))
    reward -= action_penalty

    return reward