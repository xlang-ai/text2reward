def compute_dense_reward(self, action):
    reward = 0.0

    peg_head_pose = self.peg.head_pose
    box_hole_pose = self.box.hole_pose
    peg_head_pos_at_hole = (box_hole_pose.inv() * peg_head_pose).p
    # x-axis is hole direction
    x_flag = -0.015 <= peg_head_pos_at_hole[0]
    y_flag = (-self.box.hole_radius <= peg_head_pos_at_hole[1] <= self.box.hole_radius)
    z_flag = (-self.box.hole_radius <= peg_head_pos_at_hole[2] <= self.box.hole_radius)
    success = (x_flag and y_flag and z_flag)

    if success:
        return 25.0

    # grasp pose rotation reward
    tcp_pose_wrt_peg = self.peg.pose.inv() * self.robot.ee_pose
    tcp_rot_wrt_peg = tcp_pose_wrt_peg.to_transformation_matrix()[:3, :3]
    gt_rot_1 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    gt_rot_2 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    grasp_rot_loss_fxn = lambda A: np.arcsin(
        np.clip(1 / (2 * np.sqrt(2)) * np.sqrt(np.trace(A.T @ A)), 0, 1)
    )
    grasp_rot_loss = np.minimum(
        grasp_rot_loss_fxn(gt_rot_1 - tcp_rot_wrt_peg),
        grasp_rot_loss_fxn(gt_rot_2 - tcp_rot_wrt_peg),
    ) / (np.pi / 2)
    rotated_properly = grasp_rot_loss < 0.2
    reward += 1 - grasp_rot_loss

    gripper_pos = self.robot.ee_pose.p
    tgt_gripper_pose = self.peg.pose
    offset = sapien.Pose([-0.06, 0, 0])  # account for panda gripper width with a bit more leeway
    tgt_gripper_pose = tgt_gripper_pose.transform(offset)
    if rotated_properly:
        # reaching reward
        gripper_to_peg_dist = np.linalg.norm(gripper_pos - tgt_gripper_pose.p)
        reaching_reward = 1 - np.tanh(4.0 * np.maximum(gripper_to_peg_dist - 0.015, 0.0))
        reward += reaching_reward

        # grasp reward
        is_grasped = self.robot.check_grasp(self.peg, max_angle=20)  # max_angle ensures that the gripper grasps the peg appropriately, not in a strange pose
        if is_grasped:
            reward += 2.0

        # pre-insertion award, encouraging both the peg center and the peg head to match the yz coordinates of goal_pose
        pre_inserted = False
        if is_grasped:
            peg_head_wrt_goal = self.goal_pose.inv() * self.peg.head_pose
            peg_head_wrt_goal_yz_dist = np.linalg.norm(peg_head_wrt_goal.p[1:])
            peg_wrt_goal = self.goal_pose.inv() * self.peg.pose
            peg_wrt_goal_yz_dist = np.linalg.norm(peg_wrt_goal.p[1:])
            if peg_head_wrt_goal_yz_dist < 0.01 and peg_wrt_goal_yz_dist < 0.01:
                pre_inserted = True
                reward += 3.0
            pre_insertion_reward = 3 * (1 - np.tanh(0.5 * (peg_head_wrt_goal_yz_dist + peg_wrt_goal_yz_dist) 
                    + 4.5 * np.maximum(peg_head_wrt_goal_yz_dist, peg_wrt_goal_yz_dist)))
            reward += pre_insertion_reward

        # insertion reward
        if is_grasped and pre_inserted:
            peg_head_wrt_goal_inside_hole = (self.box.hole_pose.inv() * self.peg.head_pose)
            insertion_reward = 5 * (1 - np.tanh(5.0 * np.linalg.norm(peg_head_wrt_goal_inside_hole.p)))
            reward += insertion_reward
    else:
        reward = reward - 10 * np.maximum(self.peg.pose.p[2] + self.peg.half_size[2] + 0.01 - self.robot.ee_pose.p[2], 0.0,)
        reward = reward - 10 * np.linalg.norm(tgt_gripper_pose.p[:2] - self.robot.ee_pose.p[:2])

    return reward