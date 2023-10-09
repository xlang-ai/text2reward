def compute_dense_reward(self, action):
    reward = 0.0

    # -------------------------------------------------------------------------- #
    # The end-effector should be close to the target pose
    # -------------------------------------------------------------------------- #
    handle_pose = self.cabinet.handle.pose
    ee_pose = self.robot.ee_pose

    # Position
    ee_coords = self.robot.get_ee_coords()  # [2, 10, 3]
    handle_pcd = self.cabinet.handle.get_world_pcd()

    disp_ee_to_handle = sdist.cdist(ee_coords.reshape(-1, 3), handle_pcd)
    dist_ee_to_handle = disp_ee_to_handle.reshape(2, -1).min(-1)  # [2]
    reward_ee_to_handle = -dist_ee_to_handle.mean() * 2
    reward += reward_ee_to_handle

    # Encourage grasping the handle
    ee_center_at_world = ee_coords.mean(0)  # [10, 3]
    ee_center_at_handle = transform_points(
        handle_pose.inv().to_transformation_matrix(), ee_center_at_world
    )

    dist_ee_center_to_handle = self.cabinet.handle.local_sdf(ee_center_at_handle)

    dist_ee_center_to_handle = dist_ee_center_to_handle.max()
    reward_ee_center_to_handle = (
        clip_and_normalize(dist_ee_center_to_handle, -0.01, 4e-3) - 1
    )
    reward += reward_ee_center_to_handle

    # Rotation
    target_grasp_poses = self.cabinet.handle.target_grasp_poses
    target_grasp_poses = [handle_pose * x for x in target_grasp_poses]
    angles_ee_to_grasp_poses = [angle_distance(ee_pose, x) for x in target_grasp_poses]
    ee_rot_reward = -min(angles_ee_to_grasp_poses) / np.pi * 3
    reward += ee_rot_reward

    # -------------------------------------------------------------------------- #
    # Stage reward
    # -------------------------------------------------------------------------- #
    coeff_qvel = 1.5  # joint velocity
    coeff_qpos = 0.5  # joint position distance
    stage_reward = -5 - (coeff_qvel + coeff_qpos)

    link_qpos = self.cabinet.handle.qpos
    link_qvel = self.cabinet.handle.qvel
    link_vel_norm = np.linalg.norm(self.cabinet.handle.velocity)
    link_ang_vel_norm = np.linalg.norm(self.cabinet.handle.angular_velocity)

    ee_close_to_handle = (dist_ee_to_handle.max() <= 0.01 and dist_ee_center_to_handle > 0)
    if ee_close_to_handle:
        stage_reward += 0.5

        # Distance between current and target joint positions
        reward_qpos = (clip_and_normalize(link_qpos, 0, self.cabinet.handle.target_qpos) * coeff_qpos)
        reward += reward_qpos

        if link_qpos < self.cabinet.handle.target_qpos:
            # Encourage positive joint velocity to increase joint position
            reward_qvel = clip_and_normalize(link_qvel, -0.1, 0.5) * coeff_qvel
            reward += reward_qvel
        else:
            # Add coeff_qvel for smooth transition of stagess
            stage_reward += 2 + coeff_qvel
            reward_static = -(link_vel_norm + link_ang_vel_norm * 0.5)
            reward += reward_static

            if link_vel_norm <= 0.1 and link_ang_vel_norm <= 1:
                stage_reward += 1

    reward += stage_reward
    return reward