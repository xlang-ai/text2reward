import numpy as np

def compute_dense_reward(self, action):
    reward = 0.0

    is_obj_placed = np.linalg.norm(self.goal_position - self.cubeA.pose.p) <= 0.025
    is_robot_static = np.max(np.abs(self.robot.qvel)) <= 0.2

    success = is_obj_placed and is_robot_static

    if success:
        reward += 5
        return reward

    tcp_to_obj_pos = self.cubeA.pose.p - self.robot.ee_pose.p
    tcp_to_obj_dist = np.linalg.norm(tcp_to_obj_pos)
    reaching_reward = 1 - np.tanh(5 * tcp_to_obj_dist)
    reward += reaching_reward

    is_grasped = self.robot.check_grasp(self.cubeA, max_angle=30)
    reward += 1 if is_grasped else 0.0

    if is_grasped:
        obj_to_goal_dist = np.linalg.norm(self.goal_position - self.cubeA.pose.p)
        place_reward = 1 - np.tanh(5 * obj_to_goal_dist)
        reward += place_reward

    return reward