import numpy as np

def compute_dense_reward(self, action):
    reward = 0.0

    # check if cubeA is lifted 0.2 meter
    is_obj_lifted = self.cubeA.pose.p[2] >= self.cube_half_size + self.goal_height
    # check if the robot is static
    is_robot_static = np.max(np.abs(self.robot.qvel)) <= 0.2

    # if both conditions are met, the task is successful
    success = is_obj_lifted and is_robot_static

    if success:
        reward += 5
        return reward

    # calculate the distance between robot's end-effector and cubeA
    tcp_to_obj_pos = self.cubeA.pose.p - self.robot.ee_pose.p
    tcp_to_obj_dist = np.linalg.norm(tcp_to_obj_pos)
    # calculate the reaching reward, which encourages the robot to approach cubeA
    reaching_reward = 1 - np.tanh(5 * tcp_to_obj_dist)
    reward += reaching_reward

    # check if the robot has successfully grasped cubeA
    is_grasped = self.robot.check_grasp(self.cubeA, max_angle=30)
    # if grasped, add reward
    reward += 1 if is_grasped else 0.0

    if is_grasped:
        # calculate the distance between cubeA and the target height
        obj_to_goal_dist = np.abs(self.goal_height - (self.cubeA.pose.p[2] - self.cube_half_size))
        # calculate the lifting reward, which encourages the robot to lift cubeA up
        lift_reward = 1 - np.tanh(5 * obj_to_goal_dist)
        reward += lift_reward

    return reward