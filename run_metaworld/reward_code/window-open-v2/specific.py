import numpy as np

def compute_dense_reward(self, action, obs):
    """Computes the dense reward for the Open Window task"""

    # Calculate the distance between the robot's gripper and the window's handle
    handle_dist = np.linalg.norm(obs[:3] - obs[4:7])

    # Calculate the difference between the current state of the window and its goal state
    window_diff = np.linalg.norm(obs[4:7] - self.env._get_pos_goal())

    # Regularize the robot's action
    action_reg = np.linalg.norm(action)

    # Define the weights for the components of the reward
    w1, w2, w3 = 1.0, 1.0, 0.1

    # Compute the reward as a weighted sum of the above components
    reward = -w1*handle_dist - w2*window_diff - w3*action_reg

    return reward