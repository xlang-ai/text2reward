import numpy as np

def compute_dense_reward(self, action, obs) -> float:
    # calculate the distance between the robot's gripper and the target object
    dist_to_goal = np.linalg.norm(self.robot.ee_position - self.goal_position)

    # calculate the difference between current state of object and its goal state
    state_diff = np.linalg.norm(self.obj1.position - self.goal_position)

    # regularization of the robot's action
    action_norm = np.linalg.norm(action)
    action_reg = -0.01 * action_norm

    # check if the goal is reached in y direction
    goal_reached = float(self.robot.ee_position[1] == self.goal_position[1])

    # define the weights for different parts of the reward
    dist_weight = -1
    state_diff_weight = -1
    goal_reached_weight = 100

    # calculate the total reward
    reward = dist_weight * dist_to_goal + state_diff_weight * state_diff + action_reg + goal_reached_weight * goal_reached

    return reward