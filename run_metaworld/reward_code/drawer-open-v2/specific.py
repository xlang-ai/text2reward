def compute_dense_reward(self, action, obs) -> float:
    # Define weights for different components of the reward
    w1, w2, w3, w4 = 1.0, 1.0, 0.01, 0.001 

    # Compute the distance between the robot's end effector and the handle
    dist_to_handle = np.linalg.norm(obs[:3] - obs[4:7])

    # Compute the difference between the drawer's current and goal positions
    goal_diff = np.linalg.norm(obs[4:7] - self.env._get_pos_goal())

    # Check gripper openness. If the robot is close to the handle and the gripper is not closed, penalize
    gripper_penalty = 0.0
    if dist_to_handle < 0.1 and obs[3] > -1:
        gripper_penalty = 1.0

    # Compute the regularization term for the robot's action
    action_reg = np.linalg.norm(action)

    # Compute the final reward as a weighted sum of the different components
    reward = -w1*dist_to_handle - w2*goal_diff - w3*gripper_penalty - w4*action_reg

    return reward