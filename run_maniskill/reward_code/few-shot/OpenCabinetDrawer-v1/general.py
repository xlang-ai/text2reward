from scipy.spatial import distance as sdist
import numpy as np

def compute_dense_reward(self, action: np.ndarray):
    reward = 0

    # Compute distance between end-effector and cabinet handle surface
    ee_coords = np.array(self.robot.get_ee_coords())  # [2, 10, 3]
    handle_pcd = self.cabinet.handle.get_world_pcd()  # [N, 3]

    # EE approach handle
    dist_ees_to_handle = sdist.cdist(ee_coords.reshape(-1, 3), handle_pcd)  # [20, N]
    dist_ees_to_handle = dist_ees_to_handle.min(0)  # [N]
    dist_ee_to_handle = dist_ees_to_handle.mean()
    log_dist_ee_to_handle = np.log(dist_ee_to_handle + 1e-5)
    reward += -dist_ee_to_handle - np.clip(log_dist_ee_to_handle, -10, 0)

    # Penalize action
    # Assume action is relative and normalized.
    action_norm = np.linalg.norm(action)
    reward -= action_norm * 1e-6

    # Encourage qpos change
    qpos_change = self.cabinet.handle.qpos - self.cabinet.handle.target_qpos
    reward += qpos_change * 0.1

    # Penalize the velocity of cabinet and handle
    handle_vel_norm = np.linalg.norm(self.cabinet.handle.velocity)
    reward -= handle_vel_norm * 0.01
    cabinet_vel_norm = np.linalg.norm(self.cabinet.velocity)
    reward -= cabinet_vel_norm * 0.01

    # Stage reward
    stage_reward = -10
    if dist_ee_to_handle < 0.1:
        # EE is close to handle
        stage_reward += 2
        if self.cabinet.handle.qpos >= self.cabinet.handle.target_qpos:
            # The drawer is open
            stage_reward += 8
    reward += stage_reward

    return reward