import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action):
    reward = 0.0

    # check if the handle is turned on
    is_handle_turned_on = self.current_angle >= self.target_angle
    # check if the robot is static
    is_robot_static = np.max(np.abs(self.agent.robot.get_qvel()[:-2])) <= 0.2

    success = is_handle_turned_on and is_robot_static
    
    if success:
        reward += 5
        return reward

    # calculate the reward for reaching the handle
    tcp_to_handle_pos = self.target_link.pose.p - self.tcp.pose.p
    tcp_to_handle_dist = np.linalg.norm(tcp_to_handle_pos)
    reaching_reward = 1 - np.tanh(5 * tcp_to_handle_dist)
    reward += reaching_reward

    # calculate the reward for successfully grasping the handle
    is_grasped = self.agent.check_grasp(self.target_link, max_angle=30)
    reward += 1 if is_grasped else 0.0

    if is_grasped:
        # reward for the robot turning the handle
        handle_to_target_dist = self.target_angle - self.current_angle
        turning_reward = 1 - np.tanh(5 * handle_to_target_dist)
        reward += turning_reward

    # penalize actions that could potentially damage the robot
    action_penalty = np.sum(np.square(action))
    reward -= action_penalty * 0.1

    return reward