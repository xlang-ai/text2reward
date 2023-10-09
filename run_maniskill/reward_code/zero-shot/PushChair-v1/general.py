import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action) -> float:
    # Stage 1: Encourage the robot to move towards the chair
    # Get the distance between the robot's gripper and the chair
    gripper_coords = self.robot.get_ee_coords()
    chair_pcd = self.chair.get_pcd()
    dist_to_chair = cdist(gripper_coords, chair_pcd).min(-1).mean()

    # Get the difference between the chair's current and target position
    chair_to_target_dist = np.linalg.norm(self.chair.pose.p[:2] - self.target_xy)

    # The smaller the distance, the larger the reward
    reward_dist = -dist_to_chair
    # The closer the chair is to the target, the larger the reward
    reward_target = -chair_to_target_dist

    # Stage 2: Encourage the robot to push the chair towards the target location
    # Get the velocity of the chair
    chair_vel = self.chair.velocity[:2]
    # The faster the chair moves towards the target, the larger the reward
    reward_vel = np.dot(chair_vel, (self.target_xy - self.chair.pose.p[:2])) / (np.linalg.norm(chair_vel) * chair_to_target_dist)

    # Stage 3: Prevent the chair from falling over
    # Calculate the tilt angle of the chair
    z_axis_chair = self.chair.pose.to_transformation_matrix()[:3, 2]
    chair_tilt = np.arccos(z_axis_chair[2])
    # The smaller the tilt, the larger the reward
    reward_tilt = -chair_tilt

    # Regularization of the robot's action
    reward_reg = -np.square(action).sum()

    # Weights for each stage
    w_dist = 0.2
    w_target = 0.2
    w_vel = 0.3
    w_tilt = 0.2
    w_reg = 0.1

    # Final reward
    reward = w_dist * reward_dist + w_target * reward_target + w_vel * reward_vel + w_tilt * reward_tilt + w_reg * reward_reg

    return reward