def compute_dense_reward(self, action, obs) -> float:
    # Define weight parameters for different parts of the reward
    distance_weight = 1.0
    goal_weight = 1.0
    action_weight = 0.01

    # Compute the distance between the robot's gripper and the window handle
    grip_to_handle_dist = np.linalg.norm(obs[:3] - obs[4:7])

    # Compute the difference between the window handle's current position and its goal position
    handle_to_goal_dist = np.linalg.norm(obs[4:7] - self.env._get_pos_goal())

    # Compute the regularization of the robot's action
    action_regularization = np.linalg.norm(action)

    # Compute the reward as a weighted sum of the above components
    reward = - distance_weight * grip_to_handle_dist - goal_weight * handle_to_goal_dist - action_weight * action_regularization

    # If the window handle's current position is close enough to the goal position, give a bonus reward
    if handle_to_goal_dist < 0.05:  # The threshold 0.05 can be adjusted
        reward += 1.0  # The bonus reward value 1.0 can be adjusted

    # If the robot's gripper is open when the window handle is at the goal position, give a penalty
    if handle_to_goal_dist < 0.05 and obs[3] > 0:
        reward -= 1.0  # The penalty value 1.0 can be adjusted

    return reward