from langchain.prompts import PromptTemplate

meta_prompt = """
You are an expert in robotics, reinforcement learning and code generation.
We are going to use a dual-arm mobile robot, which is a combination of one sciurus17 base and two Franka Panda arms, to complete given tasks.
The action space of the robot is a normalized `Box(-1, 1, (18,), float32)`.

Now I want you to help me write a reward function of reinforcement learning.
Typically, the reward function of a manipulation task is consisted of these following parts (some part is optional, so only include it if really necessary):
1. the distance between robot's gripper and our target object
2. difference between current state of object and its goal state
3. regularization of the robot's action
4. [optional] extra constraint of the target object, which is often implied by the task instruction
5. [optional] extra constraint of the robot, which is often implied by the task instruction
...

class BaseEnv(gym.env):
    self.robot : MobileDualArmPanda
    self.chair : ArticulateObject
    slef.bucket : ArticulateObject
    self.target_xy : np.ndarray[(2,)] # indicate the xy position of our target object position.

class MobileDualArmPanda:
    self.base_position : np.ndarray[(2,)] # indicate the xy-plane position of the Sciurus17 Mobile base
    self.base_velocity : np.ndarray[(2,)] # indicate the xy-plane velocity of the Sciurus17 Mobile base
    self.qpos : np.ndarray[(7,)] # indicate the joint position of the Franka robot
    self.qvel : np.ndarray[(7,)] # indicate the joint velocity of the Franka robot
    def get_ee_coords(self,) -> np.ndarray[(4,3)] # indicate 3D positions of 2*2=4 gripper fingers

class ObjectPose:
    self.p : np.ndarray[(3,)] # indicate the 3D position of the simple rigid object
    self.q : np.ndarray[(4,)] # indicate the quaternion of the simple rigid object
    def inv(self,) -> ObjectPose # return a `ObjectPose` class instance, which is the inverse of the original pose
    def to_transformation_matrix(self,) -> np.ndarray[(4,4)] # return a [4, 4] numpy array, indicating the transform matrix; self.to_transformation_matrix()[:3,:3] is the rotation matrix

class RigidObject:
    self.pose : ObjectPose # indicate the 3D position and quaternion of the simple rigid object
    self.velocity : np.ndarray[(3,)] # indicate the linear velocity of the simple rigid object
    self.angular_velocity : np.ndarray[(3,)] # indicate the angular velocity of the simple rigid object
    def check_static(self,) -> bool # indicate whether this rigid object is static or not

class LinkObject:
    self.pose : ObjectPose # indicate the 3D position and quaternion of the link rigid object
    self.velocity : np.ndarray[(3,)] # indicate the linear velocity of the link rigid object
    self.angular_velocity : np.ndarray[(3,)] # indicate the angular velocity of the link rigid object
    self.qpos : float # indicate the position of the link object joint
    self.qvel : float # indicate the velocity of the link object joint
    self.target_qpos : float # indicate the target position of the link object joint
    self.target_grasp_poses : list[ObjectPose] # indicate the appropriate poses for robot to grasp in the local frame
    def local_sdf(self, positions: np.ndarray[(N,3)]) -> np.ndarray[(N,)] # take in points 3D positions, and return the signed distance of these points with respect to the link object, and the input points should be transformed to the local frame of the link object first
    def get_local_pcd(self,) -> np.ndarray[(M,3)] # get the point cloud of the link object surface in the local frame
    def get_world_pcd(self,) -> np.ndarray[(M,3)] # get the point cloud of the link object surface in the world frame

class ArticulateObject:
    self.pose : ObjectPose # indicate the 3D position and quaternion of the articulated rigid object
    self.velocity : np.ndarray[(3,)] # indicate the linear velocity of the articulated rigid object
    self.angular_velocity : np.ndarray[(3,)] # indicate the angular velocity of the articulated rigid object
    self.qpos : np.ndarray[(K,)] # indicate the position of the articulated object joint
    self.qvel : np.ndarray[(K,)] # indicate the velocity of the articulated object joint
    def get_pcd(self,) -> np.ndarray[(M,3)] # indicate the point cloud of the articulated rigid object surface in the world frame
    def check_static(self,) -> bool # indicate whether the root link of this articulated object is static or not

Additional knowledge:
1. A staged reward could make the training more stable, you can write them in a nested if-else statement.
2. `ObjectPose` class support multiply operator `*`, for example: `ee_pose_wrt_cubeA = self.cubeA.pose.inv() * self.robot.end_effector.pose`
3. You can use `z_axis_object = self.object.pose.to_transformation_matrix()[:3, 2]; object_tilt = np.arccos(z_axis_object[2])`
to calculate the degree between the object and z-axis.
4. Typically, for `ArticulateObject` or `LinkObject`, you should utilize point cloud to calculate the distance between robot gripper and the object. For exmaple, you can use: `scipy.spatial.distance.cdist(pcd1, pcd2).min()` or `scipy.spatial.distance.cdist(ee_cords, pcd).min(-1).mean()`
to calculate the distance between robot gripper's 4 fingers and the nearest point on the surface of the complex articulated object.

You are allowed to use any existing python package if applicable. But only use these packages when it's really necessary.

I want it to fulfill the following task: {instruction}
1. please think step by step and tell me what does this task mean;
2. then write a function that format as `def compute_dense_reward(self, action) -> float` and returns the `reward : float` only.
3. Take care of variable's type, never use the function of another python class.
4. When you writing code, you can also add some comments as your thought, like this:
```
# TODO: Here needs to be further improved
# Here the weight of the reward is 0.5, works together with xxx's reward 0.2, and yyy's reward 0.3
# Here I define a variable called stage_reward, which is used to encourage the robot to do the task in a staged way
# Here I use the function `get_distance` to calculate the distance between the object and the target
...
```
""".strip()

MOBILE_DUAL_ARM_PROMPT_FOR_FEW_SHOT = """
You are an expert in robotics, reinforcement learning and code generation.
We are going to use a dual-arm mobile robot, which is a combination of one sciurus17 base and two Franka Panda arms, to complete given tasks.
The action space of the robot is a normalized `Box(-1, 1, (18,), float32)`.

Now I want you to help me write a reward function of reinforcement learning.
Typically, the reward function of a manipulation task is consisted of these following parts (some part is optional, so only include it if really necessary):
1. the distance between robot's gripper and our target object
2. difference between current state of object and its goal state
3. regularization of the robot's action
4. [optional] extra constraint of the target object, which is often implied by the task instruction
5. [optional] extra constraint of the robot, which is often implied by the task instruction
...

class BaseEnv(gym.env):
    self.robot : MobileDualArmPanda
    self.chair : ArticulateObject
    slef.bucket : ArticulateObject
    self.target_xy : np.ndarray[(2,)] # indicate the xy position of our target object position.

class MobileDualArmPanda:
    self.base_position : np.ndarray[(2,)] # indicate the xy-plane position of the Sciurus17 Mobile base
    self.base_velocity : np.ndarray[(2,)] # indicate the xy-plane velocity of the Sciurus17 Mobile base
    self.qpos : np.ndarray[(7,)] # indicate the joint position of the Franka robot
    self.qvel : np.ndarray[(7,)] # indicate the joint velocity of the Franka robot
    def get_ee_coords(self,) -> np.ndarray[(4,3)] # indicate 3D positions of 2*2=4 gripper fingers

class ObjectPose:
    self.p : np.ndarray[(3,)] # indicate the 3D position of the simple rigid object
    self.q : np.ndarray[(4,)] # indicate the quaternion of the simple rigid object
    def inv(self,) -> ObjectPose # return a `ObjectPose` class instance, which is the inverse of the original pose
    def to_transformation_matrix(self,) -> np.ndarray[(4,4)] # return a [4, 4] numpy array, indicating the transform matrix; self.to_transformation_matrix()[:3,:3] is the rotation matrix

class RigidObject:
    self.pose : ObjectPose # indicate the 3D position and quaternion of the simple rigid object
    self.velocity : np.ndarray[(3,)] # indicate the linear velocity of the simple rigid object
    self.angular_velocity : np.ndarray[(3,)] # indicate the angular velocity of the simple rigid object
    def check_static(self,) -> bool # indicate whether this rigid object is static or not

class LinkObject:
    self.pose : ObjectPose # indicate the 3D position and quaternion of the link rigid object
    self.velocity : np.ndarray[(3,)] # indicate the linear velocity of the link rigid object
    self.angular_velocity : np.ndarray[(3,)] # indicate the angular velocity of the link rigid object
    self.qpos : float # indicate the position of the link object joint
    self.qvel : float # indicate the velocity of the link object joint
    self.target_qpos : float # indicate the target position of the link object joint
    self.target_grasp_poses : list[ObjectPose] # indicate the appropriate poses for robot to grasp in the local frame
    def local_sdf(self, positions: np.ndarray[(N,3)]) -> np.ndarray[(N,)] # take in points 3D positions, and return the signed distance of these points with respect to the link object, and the input points should be transformed to the local frame of the link object first
    def get_local_pcd(self,) -> np.ndarray[(M,3)] # get the point cloud of the link object surface in the local frame
    def get_world_pcd(self,) -> np.ndarray[(M,3)] # get the point cloud of the link object surface in the world frame

class ArticulateObject:
    self.pose : ObjectPose # indicate the 3D position and quaternion of the articulated rigid object
    self.velocity : np.ndarray[(3,)] # indicate the linear velocity of the articulated rigid object
    self.angular_velocity : np.ndarray[(3,)] # indicate the angular velocity of the articulated rigid object
    self.qpos : np.ndarray[(K,)] # indicate the position of the articulated object joint
    self.qvel : np.ndarray[(K,)] # indicate the velocity of the articulated object joint
    def get_pcd(self,) -> np.ndarray[(M,3)] # indicate the point cloud of the articulated rigid object surface in the world frame
    def check_static(self,) -> bool # indicate whether the root link of this articulated object is static or not

Additional knowledge:
1. A staged reward could make the training more stable, you can write them in a nested if-else statement.
2. `ObjectPose` class support multiply operator `*`, for example: `ee_pose_wrt_cubeA = self.cubeA.pose.inv() * self.robot.end_effector.pose`
3. You can use `z_axis_object = self.object.pose.to_transformation_matrix()[:3, 2]; object_tilt = np.arccos(z_axis_object[2])`
to calculate the degree between the object and z-axis.
4. Typically, for `ArticulateObject` or `LinkObject`, you should utilize point cloud to calculate the distance between robot gripper and the object. For exmaple, you can use: `scipy.spatial.distance.cdist(pcd1, pcd2).min()` or `scipy.spatial.distance.cdist(ee_cords, pcd).min(-1).mean()`
to calculate the distance between robot gripper's 4 fingers and the nearest point on the surface of the complex articulated object.

You are allowed to use any existing python package if applicable. But only use these packages when it's really necessary.

I want it to fulfill certain task, here are some tips, tricks and examples:
1. please think step by step and tell me what does this task mean;
2. then write a function that format as `def compute_dense_reward(self, action) -> float` and returns the `reward : float` only.
3. Take care of variable's type, never use the function of another python class.
4. When you writing code, you can also add some comments as your thought, like this:
```
# TODO: Here needs to be further improved
# Here the weight of the reward is 0.5, works together with xxx's reward 0.2, and yyy's reward 0.3
# Here I define a variable called stage_reward, which is used to encourage the robot to do the task in a staged way
# Here I use the function `get_distance` to calculate the distance between the object and the target
...
```
""".strip()

MOBILE_DUAL_ARM_PROMPT = PromptTemplate(
    input_variables=["instruction"],
    template=meta_prompt
)
