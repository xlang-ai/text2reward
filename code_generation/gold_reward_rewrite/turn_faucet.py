def compute_dense_reward(self, action):
    reward = 0.0

    angle_diff = self.faucet.handle.target_qpos - self.faucet.handle.qpos
    success = angle_diff < 0

    if success:
        return 10.0

    """Compute the distance between the tap and robot fingers."""
    pcd = self.faucet.handle.get_world_pcd()
    pcd1 = self.robot.lfinger.get_world_pcd()
    pcd2 = self.robot.rfinger.get_world_pcd()

    distance1 = cdist(pcd, pcd1)
    distance2 = cdist(pcd, pcd2)

    distance = min(distance1.min(), distance2.min())

    reward += 1 - np.tanh(distance * 5.0)

    turn_reward_1 = 3 * (1 - np.tanh(max(angle_diff, 0) * 2.0))
    reward += turn_reward_1

    delta_angle = angle_diff - self.last_angle_diff
    if angle_diff > 0:
        turn_reward_2 = -np.tanh(delta_angle * 2)
    else:
        turn_reward_2 = np.tanh(delta_angle * 2)
    turn_reward_2 *= 5
    reward += turn_reward_2

    self.last_angle_diff = angle_diff

    return reward