import numpy as np
from scipy.spatial.transform import Rotation as R

def compute_dense_reward(self, action: np.ndarray, obs) -> float:
    # Distance between robot's gripper and the drawer's handle
    gripper_handle_dist = np.linalg.norm(self.robot.ee_position - self.obj1.position)

    # Difference between current state of drawer's position and its goal state
    drawer_goal_dist = np.linalg.norm(self.obj1.position - self.goal_position)

    # The gripper should be able to grasp the handle. We can encourage this by
    # adding a reward component that takes into account the gripper's openness.
    # Assuming that a gripper_openness of -1 means fully open, and 1 means fully closed.
    gripper_reward = -abs(self.robot.gripper_openness + 1)  # Maximum reward when fully open.

    # Regularization of the robot's action: we want the robot to perform the task
    # with minimal and efficient action. A common way to achieve this is to penalize
    # large actions.
    action_reg = -np.linalg.norm(action)

    # Weights for each reward component
    w1, w2, w3, w4 = 1.0, 1.0, 0.5, 0.1

    # Combine the reward components
    reward = w1 * (-gripper_handle_dist) + w2 * (-drawer_goal_dist) + w3 * gripper_reward + w4 * action_reg

    return reward