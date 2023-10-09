import numpy as np
import scipy.spatial.distance as sdist

def compute_dense_reward(self, action):
    reward = -20.0

    actor = self.root_link
    ee_coords = np.array(self.agent.get_ee_coords())
    ee_mids = np.array([ee_coords[:2].mean(0), ee_coords[2:].mean(0)])
    chair_pcd = self.env.env._get_chair_pcd()

    # EE approach chair
    dist_ees_to_chair = sdist.cdist(ee_coords, chair_pcd)  # [4, N]
    dist_ees_to_chair = dist_ees_to_chair.min(1)  # [4]
    dist_ee_to_chair = dist_ees_to_chair.mean()
    log_dist_ee_to_chair = np.log(dist_ee_to_chair + 1e-5)
    reward += -dist_ee_to_chair - np.clip(log_dist_ee_to_chair, -10, 0)

    # Penalize action
    action_norm = np.linalg.norm(action)
    reward -= action_norm * 1e-6

    # Keep chair standing
    z_axis_world = np.array([0, 0, 1])
    z_axis_chair = self.root_link.pose.to_transformation_matrix()[:3, 2]
    chair_tilt = np.arccos(z_axis_chair[2])
    log_chair_tilt = np.log(chair_tilt + 1e-5)
    reward += -chair_tilt * 0.2

    # Chair velocity
    chair_vel = actor.velocity
    chair_vel_norm = np.linalg.norm(chair_vel)
    disp_chair_to_target = self.root_link.pose.p[:2] - self.target_xy
    chair_vel_dir = sdist.cosine(chair_vel[:2], disp_chair_to_target)
    chair_ang_vel_norm = np.linalg.norm(actor.angular_velocity)

    # Stage reward
    stage_reward = 0

    dist_chair_to_target = np.linalg.norm(self.root_link.pose.p[:2] - self.target_xy)

    if dist_ee_to_chair < 0.2:
        stage_reward += 2
        if dist_chair_to_target <= 0.3:
            stage_reward += 2
            reward += (np.exp(-chair_vel_norm * 10) * 2)
            if chair_vel_norm <= 0.1 and chair_ang_vel_norm <= 0.2:
                stage_reward += 2
                if chair_tilt <= 0.1 * np.pi:
                    stage_reward += 2
        else:
            reward_vel = (chair_vel_dir - 1) * chair_vel_norm
            reward += np.clip(1 - np.exp(-reward_vel), -1, np.inf) * 2 - dist_chair_to_target * 2

    if chair_tilt > 0.4 * np.pi:
        stage_reward -= 2

    reward = reward + stage_reward
    return reward