import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action):
    # Here I define the reward and penalty weights for different aspects of the task
    handle_reach_weight = 0.1
    grasp_handle_weight = 0.2
    rotation_weight = 0.7
    action_penalty_weight = 0.01

    # Here I define the grasp success reward
    grasp_success_reward = 0.1

    # Here I compute the distance between the robot's end effector and the faucet handle
    handle_pcd = transform_points(self.target_link.pose.to_transformation_matrix(), self.target_link_pcd)
    lfinger_cords = np.array([self.lfinger.pose.p, ])
    rfinger_cords = np.array([self.rfinger.pose.p, ])
    dist_lfinger_handle = cdist(lfinger_cords, handle_pcd).min(-1)[0]
    dist_rfinger_handle = cdist(rfinger_cords, handle_pcd).min(-1)[0]
    dist_handle_reach = max(dist_lfinger_handle, dist_rfinger_handle)
    handle_reach_reward = - handle_reach_weight * dist_handle_reach

    # Here I check if the robot has successfully grasped the faucet handle
    grasp_handle = self.agent.check_grasp(self.target_link)
    grasp_handle_reward = grasp_handle_weight * grasp_handle if grasp_handle else 0

    # Here I calculate the rotation reward based on the difference between the current and target joint position
    rotation_diff = max(0, self.target_angle - self.current_angle)
    rotation_reward = - rotation_weight * rotation_diff

    # Here I calculate the penalty for the robot's action
    action_penalty = - action_penalty_weight * np.linalg.norm(action)

    # The total reward is the sum of all the individual rewards and penalties
    reward = handle_reach_reward + grasp_handle_reward + rotation_reward + action_penalty

    if grasp_handle and rotation_diff == 0:
        reward += grasp_success_reward

    return reward