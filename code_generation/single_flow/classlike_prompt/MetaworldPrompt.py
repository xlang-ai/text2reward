from langchain.prompts import PromptTemplate

classlike_prompt = """
You are an expert in robotics, reinforcement learning and code generation.
We are going to use a robot arm to complete given tasks. The action space of the robot is a normalized `Box(-1, 1, (4,), float32)`.

Now I want you to help me write a reward function of reinforcement learning.
Typically, the reward function of a manipulation task is consisted of these following parts (some part is optional, so only include it if really necessary):
1. the distance between robot's gripper and our target object
2. difference between current state of object and its goal state
3. regularization of the robot's action
...

class BaseEnv(gym.env):
    self.robot : Robot # the robot in the environment
    self.obj1 : RigidObject # the first object in the environment
    self.obj2 : RigidObject # the second object in the environment, **if any**
    self.goal_position : np.ndarray[(3,)] # indicate the 3D position of the goal

class Robot:
    self.ee_position : np.ndarray[(3,)] # indicate the 3D position of the end-effector
    self.gripper_openness : float # a normalized measurement of how open the gripper is, range in [-1, 1]

class RigidObject:
    self.position : np.ndarray[(3,)] # indicate the 3D position of the rigid object
    self.quaternion : np.ndarray[(4,)] # indicate the quaternion of the rigid object

You are allowed to use any existing python package if applicable. But only use these packages when it's really necessary.

I want it to fulfill the following task: {instruction}
1. Please think step by step and tell me what does this task mean;
2. Then write a function that format as `def compute_dense_reward(self, action, obs) -> float` and returns the reward. Just the function body is fine.
3. Do not invent any variable or attribute that is not given.
4. When you writing code, you can also add some comments as your thought.
""".strip()


METAWORLD_PROMPT = PromptTemplate(
    input_variables=["instruction"],
    template=classlike_prompt
)