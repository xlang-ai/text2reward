import numpy as np
from scipy.spatial import distance

def compute_dense_reward(self, action) -> float:
    # Define reward weights
    weight_dist = 0.4
    weight_lift = 0.4
    weight_grasp = 0.2

    # Initialize reward
    reward = 0.0

    # Stage 1: Approach the cube
    ee_pos = self.robot.ee_pose.p
    cube_pos = self.cubeA.pose.p
    dist_to_cube = np.linalg.norm(ee_pos - cube_pos)
    reward_dist = -weight_dist * dist_to_cube
    
    # Stage 2: Grasp the cube
    grasp_success = self.robot.check_grasp(self.cubeA, max_angle=30)
    reward_grasp = weight_grasp * grasp_success

    # Stage 3: Lift the cube
    lift_amount = cube_pos[2] - self.goal_height
    reward_lift = -weight_lift * np.abs(lift_amount)

    # Total reward
    reward = reward_dist + reward_grasp + reward_lift

    # Stage 4: Maintain the cube at the goal height
    if self.cubeA.pose.p[2] >= self.goal_height:
        reward += 0.1 * (self.cubeA.pose.p[2] - self.goal_height)
        
    # Regularize the robot's action
    # We don't want robot to take very big action, so we add a negative reward here
    reward -= 0.01 * np.linalg.norm(action)

    return reward