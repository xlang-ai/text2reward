def compute_dense_reward(self, action):
    reward = 0.0

    is_obj_placed = self.cubeA.pose.p[2] >= self.goal_height + self.cube_half_size
    is_robot_static = np.max(np.abs(self.robot.qvel)) <= 0.2

    success = is_obj_placed and is_robot_static

    if success:
        reward += 2.25
        return reward

    # reaching reward
    gripper_pos = self.robot.ee_pose.p
    obj_pos = self.cubeA.pose.p
    dist = np.linalg.norm(gripper_pos - obj_pos)
    reaching_reward = 1 - np.tanh(5 * dist)
    reward += reaching_reward

    is_grasped = self.robot.check_grasp(self.cubeA, max_angle=30)

    # grasp reward
    if is_grasped:
        reward += 0.25

    # lifting reward
    if is_grasped:
        lifting_reward = self.cubeA.pose.p[2] - self.cube_half_size
        lifting_reward = min(lifting_reward / self.goal_height, 1.0)
        reward += lifting_reward

    return reward