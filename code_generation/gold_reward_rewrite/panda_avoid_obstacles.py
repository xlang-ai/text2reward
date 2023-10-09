def compute_dense_reward(self, action):
    tcp_pose_at_goal = self.goal_pose.inv() * self.robot.ee_pose
    pos_dist = np.linalg.norm(tcp_pose_at_goal.p)
    ang_dist = np.arccos(tcp_pose_at_goal.q[0]) * 2
    if ang_dist > np.pi:  # [0, 2 * pi] -> [-pi, pi]
        ang_dist = ang_dist - 2 * np.pi
    ang_dist = np.abs(ang_dist)
    ang_dist = np.rad2deg(ang_dist)
    success = pos_dist <= 0.025 and ang_dist <= 15

    if success:
        return 10.0

    pos_threshold = 0.025
    ang_threshold = 15
    reward = 0.0
    num_obstacles = len(self.obstacles)

    close_to_goal_reward = (
        4.0 * np.sum(pos_dist < self.goal_to_obstacle_dist) / num_obstacles
    )
    angular_reward = 0.0

    smallest_g2o_dist = self.goal_to_obstacle_dist[0]
    if pos_dist < smallest_g2o_dist:
        angular_reward = 3.0 * (1 - np.tanh(np.maximum(ang_dist - ang_threshold, 0.0) / 180))
        if ang_dist <= 25:
            close_to_goal_reward += 2.0 * (1- np.tanh(np.maximum(pos_dist - pos_threshold, 0.0) / smallest_g2o_dist))

    max_impulse_norm = self.robot.get_max_impulse_norm()
    reward = close_to_goal_reward + angular_reward - 50.0 * max_impulse_norm
    return reward