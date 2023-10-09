from scipy.spatial import distance as sdist

def compute_dense_reward(self, action: np.ndarray) -> float:
    reward = 0

    # Compute distance from end-effector to cabinet door handle
    ee_coords = self.agent.get_ee_coords()  # [2, 3]
    handle_pcd = transform_points(self.target_link.pose.to_transformation_matrix(), self.target_handle_pcd)  # [N, 3]
    
    # EE approach handle
    dist_ee_to_handle = sdist.cdist(ee_coords, handle_pcd)  # [2, N]
    dist_ee_to_handle = dist_ee_to_handle.min(1)  # [2]
    dist_ee_to_handle = dist_ee_to_handle.mean()
    log_dist_ee_to_handle = np.log(dist_ee_to_handle + 1e-5)
    reward += -dist_ee_to_handle - np.clip(log_dist_ee_to_handle, -10, 0)

    # Penalize action
    # Assume action is relative and normalized.
    action_norm = np.linalg.norm(action)
    reward -= action_norm * 1e-6

    # Cabinet door position
    cabinet_door_pos = self.link_qpos
    target_door_pos = self.target_qpos

    # Stage reward
    stage_reward = -10
    if cabinet_door_pos > target_door_pos:
        # Cabinet door is opened
        stage_reward += 5
    else:
        # Encourage the robot to continue moving the cabinet door
        stage_reward += (cabinet_door_pos - target_door_pos) * 2

    reward = reward + stage_reward
    return reward