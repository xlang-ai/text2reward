def compute_dense_reward(self, action):
    reward = -20.0

    actor = self.bucket
    ee_coords = np.array(self.robot.get_ee_coords())
    ee_mids = np.array([ee_coords[:2].mean(0), ee_coords[2:].mean(0)])
    bucket_pcd = self.bucket.get_world_pcd()

    # EE approach bucket
    dist_ees_to_bucket = sdist.cdist(ee_coords, bucket_pcd)  # [4, N]
    dist_ees_to_bucket = dist_ees_to_bucket.min(1)  # [4]
    dist_ee_to_bucket = dist_ees_to_bucket.mean()
    log_dist_ee_to_bucket = np.log(dist_ee_to_bucket + 1e-5)
    reward += -dist_ee_to_bucket - np.clip(log_dist_ee_to_bucket, -10, 0)

    # EE adjust height
    bucket_mid = self.bucket.body_link.cmass_pose.p
    bucket_mid[2] += self.bucket_center_offset
    v1 = ee_mids[0] - bucket_mid
    v2 = ee_mids[1] - bucket_mid
    ees_oppo = sdist.cosine(v1, v2)
    ees_height_diff = abs((quat2mat(self.bucket.pose.q).T @ (ee_mids[0] - ee_mids[1]))[2])
    log_ees_height_diff = np.log(ees_height_diff + 1e-5)
    reward += -np.clip(log_ees_height_diff, -10, 0) * 0.2

    # Keep bucket standing
    z_axis_world = np.array([0, 0, 1])
    z_axis_bucket = quat2mat(self.bucket.pose.q) @ z_axis_world
    bucket_tilt = abs(angle_between_vec(z_axis_world, z_axis_bucket))
    log_dist_ori = np.log(bucket_tilt + 1e-5)
    reward += -bucket_tilt * 0.2

    # Penalize action
    # Assume action is relative and normalized.
    action_norm = np.linalg.norm(action)
    reward -= action_norm * 1e-6

    # Bucket velocity
    actor_vel = actor.velocity
    actor_vel_norm = np.linalg.norm(actor_vel)
    disp_bucket_to_target = self.bucket.pose.p[:2] - self.target_xy
    actor_vel_dir = sdist.cosine(actor_vel[:2], disp_bucket_to_target)
    actor_ang_vel_norm = np.linalg.norm(actor.angular_velocity)
    actor_vel_up = actor_vel[2]

    # Stage reward
    stage_reward = 0

    bucket_height = self.bucket.body_link.cmass_pose.p[2]
    dist_bucket_height = np.linalg.norm(bucket_height - self.init_bucket_height - 0.2)
    dist_bucket_to_target = np.linalg.norm(self.bucket.pose.p[:2] - self.target_xy)

    if dist_ee_to_bucket < 0.1:
        stage_reward += 2
        reward += ees_oppo * 2

        bucket_height = self.bucket.body_link.cmass_pose.p[2]
        
        dist_bucket_height = np.linalg.norm(
            bucket_height - self.init_bucket_height - 0.2
        )
        if dist_bucket_height < 0.03:
            stage_reward += 2
            reward -= np.clip(log_dist_ori, -4, 0)
            if dist_bucket_to_target <= 0.3:
                stage_reward += 2
                reward += (np.exp(-actor_vel_norm * 10) * 2)
                if actor_vel_norm <= 0.1 and actor_ang_vel_norm <= 0.2:
                    stage_reward += 2
                    if bucket_tilt <= 0.1 * np.pi:
                        stage_reward += 2
            else:
                reward_vel = (actor_vel_dir - 1) * actor_vel_norm
                reward += (
                    np.clip(1 - np.exp(-reward_vel), -1, np.inf) * 2
                    - dist_bucket_to_target * 2
                )
        else:
            reward += (
                np.clip(1 - np.exp(-actor_vel_up), -1, np.inf) * 2
                - dist_bucket_height * 20
            )

    if bucket_tilt > 0.4 * np.pi:
        stage_reward -= 2

    reward = reward + stage_reward
    return reward