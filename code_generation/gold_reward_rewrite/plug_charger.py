def compute_dense_reward(self, action):
    reward = 0.0

    obj_pose = self.charger.pose
    obj_to_goal_pos = self.goal_pose.p - obj_pose.p
    obj_to_goal_dist = np.linalg.norm(obj_to_goal_pos)

    obj_to_goal_quat = qmult(qinverse(self.goal_pose.q), obj_pose.q)
    _, obj_to_goal_angle = quat2axangle(obj_to_goal_quat)
    obj_to_goal_angle = min(obj_to_goal_angle, np.pi * 2 - obj_to_goal_angle)

    success = obj_to_goal_dist <= 5e-3 and obj_to_goal_angle <= 0.2

    if success:
        return 50.0

    # grasp pose rotation reward
    tcp_pose_wrt_charger = self.charger.cmass_pose.inv() * self.robot.ee_pose
    tcp_rot_wrt_charger = tcp_pose_wrt_charger.to_transformation_matrix()[:3, :3]
    gt_rot_1 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    gt_rot_2 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    grasp_rot_loss_fxn = lambda A: np.tanh(
        1 / 4 * np.trace(A.T @ A)
    )  # trace(A.T @ A) has range [0,8] for A being difference of rotation matrices
    grasp_rot_loss = np.minimum(
        grasp_rot_loss_fxn(gt_rot_1 - tcp_rot_wrt_charger),
        grasp_rot_loss_fxn(gt_rot_2 - tcp_rot_wrt_charger),
    )
    reward += 2 * (1 - grasp_rot_loss)
    rotated_properly = grasp_rot_loss < 0.1

    if rotated_properly:
        # reaching reward
        gripper_to_obj_pos = self.charger.cmass_pose.p - self.robot.ee_pose.p
        gripper_to_obj_dist = np.linalg.norm(gripper_to_obj_pos)
        reaching_reward = 1 - np.tanh(5.0 * gripper_to_obj_dist)
        reward += 2 * reaching_reward

        # grasp reward
        is_grasped = self.robot.check_grasp(
            self.charger, max_angle=20
        )  # max_angle ensures that the gripper grasps the charger appropriately, not in a strange pose
        if is_grasped:
            reward += 2.0

        # pre-insertion and insertion award
        if is_grasped:
            pre_inserted = False
            charger_cmass_wrt_goal = self.goal_pose.inv() * self.charger.cmass_pose
            charger_cmass_wrt_goal_yz_dist = np.linalg.norm(charger_cmass_wrt_goal.p[1:])
            charger_cmass_wrt_goal_rot = (charger_cmass_wrt_goal.to_transformation_matrix()[:3, :3])
            charger_wrt_goal = self.goal_pose.inv() * self.charger.pose
            charger_wrt_goal_yz_dist = np.linalg.norm(charger_wrt_goal.p[1:])
            charger_wrt_goal_dist = np.linalg.norm(charger_wrt_goal.p)
            charger_wrt_goal_rot = (charger_cmass_wrt_goal.to_transformation_matrix()[:3, :3])

            gt_rot = np.eye(3)
            rot_loss_fxn = lambda A: np.tanh(1 / 2 * np.trace(A.T @ A))
            rot_loss = np.maximum(rot_loss_fxn(charger_cmass_wrt_goal_rot - gt_rot), rot_loss_fxn(charger_wrt_goal_rot - gt_rot))

            pre_insertion_reward = 3 * (1 - np.tanh(1.0 * (charger_cmass_wrt_goal_yz_dist + charger_wrt_goal_yz_dist) + 
                                                    9.0 * np.maximum(charger_cmass_wrt_goal_yz_dist, charger_wrt_goal_yz_dist)))
            pre_insertion_reward += 3 * (1 - np.tanh(3 * charger_wrt_goal_dist))
            pre_insertion_reward += 3 * (1 - rot_loss)
            reward += pre_insertion_reward

            if (
                charger_cmass_wrt_goal_yz_dist < 0.01
                and charger_wrt_goal_yz_dist < 0.01
                and charger_wrt_goal_dist < 0.02
                and rot_loss < 0.15
            ):
                pre_inserted = True
                reward += 2.0

            if pre_inserted:
                insertion_reward = 2 * (1 - np.tanh(25.0 * charger_wrt_goal_dist))
                insertion_reward += 5 * (
                    1 - np.tanh(2.0 * np.abs(obj_to_goal_angle))
                )
                insertion_reward += 5 * (1 - rot_loss)
                reward += insertion_reward
    else:
        reward = reward - 10 * np.maximum(
            self.charger.pose.p[2]
            + self.charger.base_half_size[2] / 2
            + 0.015
            - self.robot.ee_pose.p[2],
            0.0)
        reward = reward - 10 * np.linalg.norm(self.charger.pose.p[:2] - self.robot.ee_pose.p[:2])

    return reward