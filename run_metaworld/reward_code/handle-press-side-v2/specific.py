def compute_dense_reward(self, action, obs) -> float:
    # Constants to weight various components of the reward
    distance_weight = 1.0
    state_difference_weight = 1.0
    action_regularization_weight = 0.1

    # Computing the Euclidean distance between the gripper and the handle
    gripper_handle_distance = np.linalg.norm(obs[:3] - obs[4:7])
    # Normalizing the distance to be within the range -1 to 1
    gripper_handle_distance = np.clip(gripper_handle_distance, -1, 1)

    # Computing the difference between the current and goal states of the handle
    state_difference = np.linalg.norm(obs[4:7] - self.env._get_pos_goal())
    # Normalizing the difference to be within the range -1 to 1
    state_difference = np.clip(state_difference, -1, 1)

    # Computing the regularization of the robot's action
    action_regularization = np.linalg.norm(action)
    # Normalizing the regularization to be within the range -1 to 1
    action_regularization = np.clip(action_regularization, -1, 1)

    # Weighting and summing the components to compute the final reward
    reward = -distance_weight * gripper_handle_distance \
             -state_difference_weight * state_difference \
             -action_regularization_weight * action_regularization

    return reward