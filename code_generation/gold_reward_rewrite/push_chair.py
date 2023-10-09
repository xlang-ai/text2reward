def compute_dense_reward(self, action: np.ndarray):
        reward = 0

        # Compute distance between end-effectors and chair surface
        ee_coords = np.array(self.robot.get_ee_coords())  # [4, 3]
        chair_pcd = self.chair.get_world_pcd()  # [N, 3]

        # EE approach chair
        dist_ees_to_chair = sdist.cdist(ee_coords, chair_pcd)  # [4, N]
        dist_ees_to_chair = dist_ees_to_chair.min(1)  # [4]
        dist_ee_to_chair = dist_ees_to_chair.mean()
        log_dist_ee_to_chair = np.log(dist_ee_to_chair + 1e-5)
        reward += -dist_ee_to_chair - np.clip(log_dist_ee_to_chair, -10, 0)

        # Keep chair standing
        # z-axis of chair should be upward
        z_axis_chair = self.chair.pose.to_transformation_matrix()[:3, 2]
        chair_tilt = np.arccos(z_axis_chair[2])
        reward += -chair_tilt * 0.2

        # Penalize action
        # Assume action is relative and normalized.
        action_norm = np.linalg.norm(action)
        reward -= action_norm * 1e-6

        # Chair velocity
        chair_vel = self.chair.velocity[:2]
        chair_vel_norm = np.linalg.norm(chair_vel)
        disp_chair_to_target = self.chair.pose.p[:2] - self.target_xy
        cos_chair_vel_to_target = sdist.cosine(disp_chair_to_target, chair_vel)
        chair_ang_vel_norm = np.linalg.norm(self.chair.angular_velocity)

        # Stage reward
        stage_reward = -10
        disp_chair_to_target = self.chair.pose.p[:2] - self.target_xy
        dist_chair_to_target = np.linalg.norm(disp_chair_to_target)

        if chair_tilt < 0.2 * np.pi:
            # Chair is standing
            if dist_ee_to_chair < 0.1:
                # EE is close to chair
                stage_reward += 2
                if dist_chair_to_target <= 0.15:
                    # Chair is close to target
                    stage_reward += 2
                    # Try to keep chair static
                    reward += np.exp(-chair_vel_norm * 10) * 2
                    if chair_vel_norm <= 0.1 and chair_ang_vel_norm <= 0.2:
                        stage_reward += 2
                else:
                    # Try to increase velocity along direction to the target
                    # Compute directional velocity
                    x = (1 - cos_chair_vel_to_target) * chair_vel_norm
                    reward += max(-1, 1 - np.exp(x)) * 2 - dist_chair_to_target * 2
        else:
            stage_reward = -5

        reward = reward + stage_reward
        return reward