import numpy as np

def compute_dense_reward(self, action):
    reward = 0.0

    pos_A = self.cubeA.pose.p
    pos_B = self.cubeB.pose.p
    offset = pos_A - pos_B
    xy_flag = (np.linalg.norm(offset[:2]) <= self.cube_half_size + 0.005)
    z_flag = np.abs(offset[2] - self.cube_half_size * 2) <= 0.005

    is_cubeA_on_cubeB = bool(xy_flag and z_flag)
    is_cubeA_static = self.cubeA.check_static()
    is_cubaA_grasped = self.robot.check_grasp(self.cubeA)
    success = is_cubeA_on_cubeB and is_cubeA_static and (not is_cubaA_grasped)

    if success:
        reward = 15.0
    else:
        # grasp pose rotation reward
        grasp_rot_loss_fxn = lambda A: np.tanh(
            1 / 8 * np.trace(A.T @ A)
        )  # trace(A.T @ A) has range [0,8] for A being difference of rotation matrices
        tcp_pose_wrt_cubeA = self.cubeA.pose.inv() * self.robot.ee_pose
        tcp_rot_wrt_cubeA = tcp_pose_wrt_cubeA.to_transformation_matrix()[:3, :3]
        gt_rots = [
            np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]),
            np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]),
            np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
            np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
        ]
        grasp_rot_loss = min([grasp_rot_loss_fxn(x - tcp_rot_wrt_cubeA) for x in gt_rots])
        reward += 1 - grasp_rot_loss

        cubeB_vel_penalty = np.linalg.norm(self.cubeB.velocity) + np.linalg.norm(self.cubeB.angular_velocity)
        reward -= cubeB_vel_penalty

        # reaching object reward
        tcp_pose = self.robot.ee_pose.p
        cubeA_pos = self.cubeA.pose.p
        cubeA_to_tcp_dist = np.linalg.norm(tcp_pose - cubeA_pos)
        reaching_reward = 1 - np.tanh(3.0 * cubeA_to_tcp_dist)
        reward += reaching_reward

        # check if cubeA is on cubeB
        cubeA_pos = self.cubeA.pose.p
        cubeB_pos = self.cubeB.pose.p
        goal_xyz = np.hstack([cubeB_pos[0:2], cubeB_pos[2] + self.cube_half_size * 2])
        cubeA_on_cubeB = (np.linalg.norm(goal_xyz[:2] - cubeA_pos[:2]) < self.cube_half_size * 0.8)
        cubeA_on_cubeB = cubeA_on_cubeB and (np.abs(goal_xyz[2] - cubeA_pos[2]) <= 0.005)
        if cubeA_on_cubeB:
            reward = 10.0
            # ungrasp reward
            is_cubeA_grasped = self.robot.check_grasp(self.cubeA)
            if not is_cubeA_grasped:
                reward += 2.0
            else:
                reward += 2.0 * gripper_openness
        else:
            # grasping reward
            is_cubeA_grasped = self.robot.check_grasp(self.cubeA)
            if is_cubeA_grasped:
                reward += 1.0

            # reaching goal reward, ensuring that cubeA has appropriate height during this process
            if is_cubeA_grasped:
                cubeA_to_goal = goal_xyz - cubeA_pos
                cubeA_to_goal_dist = np.linalg.norm(cubeA_to_goal)
                appropriate_height_penalty = np.maximum(
                    np.maximum(2 * cubeA_to_goal[2], 0.0),
                    np.maximum(2 * (-0.02 - cubeA_to_goal[2]), 0.0),
                )
                reaching_reward2 = 2 * (1 - np.tanh(5.0 * appropriate_height_penalty))
                reaching_reward2 += 4 * (1 - np.tanh(5.0 * cubeA_to_goal_dist))
                reward += np.maximum(reaching_reward2, 0.0)

    return reward