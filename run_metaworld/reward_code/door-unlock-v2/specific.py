import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action, obs) -> float:
    # Define constants for reward tuning
    DISTANCE_WEIGHT = 1.0
    GOAL_REACHED_REWARD = 100.0
    ACTION_PENALTY = 0.1

    # Compute distance between robot's gripper and the lock
    distance = np.linalg.norm(obs[:3] - obs[4:7])

    # Compute difference between current state of object and its goal state
    goal_diff = np.linalg.norm(obs[4:7] - self.env._get_pos_goal())

    # Compute action regularization term
    action_penalty = ACTION_PENALTY * np.square(action).sum()

    # Check if the goal has been reached
    goal_reached = cdist(obs[4:7].reshape(1, -1), self.env._get_pos_goal().reshape(1, -1), 'cosine') < 0.01

    # Calculate reward
    reward = - DISTANCE_WEIGHT * distance - goal_diff - action_penalty
    if goal_reached:
        reward += GOAL_REACHED_REWARD

    return reward