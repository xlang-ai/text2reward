import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action, obs) -> float:
    # Define constants for reward tuning
    DISTANCE_WEIGHT = 1.0
    GOAL_REACHED_REWARD = 100.0
    ACTION_PENALTY = 0.1

    # Compute distance between robot's gripper and the lock
    distance = np.linalg.norm(self.robot.ee_position - self.obj1.position)

    # Compute difference between current state of object and its goal state
    goal_diff = np.linalg.norm(self.obj1.position - self.goal_position)

    # Compute action regularization term
    action_penalty = ACTION_PENALTY * np.square(action).sum()

    # Check if the goal has been reached
    goal_reached = cdist(self.obj1.position.reshape(1, -1), self.goal_position.reshape(1, -1), 'cosine') < 0.01

    # Calculate reward
    reward = - DISTANCE_WEIGHT * distance - goal_diff - action_penalty
    if goal_reached:
        reward += GOAL_REACHED_REWARD

    return reward