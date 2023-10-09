from langchain.prompts import PromptTemplate

classlike_prompt = """
You are an expert in robotics locomotion, reinforcement learning and code generation.
We are going to control a Hopper to complete given tasks. 
The hopper is a two-dimensional one-legged figure that consist of four main body parts: the torso at the top, the thigh in the middle, the leg in the bottom, and a single foot on which the entire body rests. All adjacent body parts are connected by hinges.
The action space of the Hopper is a Box(-1, 1, (3,), float32), with action[0] indiactes Torque applied on the thigh rotor, action[1] indiactes Torque applied on the leg rotor, and action[2] indiactes Torque applied on the foot rotor

Now I want you to help me write a reward function of reinforcement learning.
I'll give you the attributes of the environment and Hopper itself. You can use these class attributes to write the reward function.

class BaseEnv(gym.Env):
    self.hopper : HopperRobot
    self.dt : float # the time between two actions, in seconds, default is 0.008 s
    def __init__(self): # initialzie state of the Hopper
        self.hopper.top.position_x = 0.0
        self.hopper.top.position_z = 1.25
        self.hopper.top_joint.angle = 0.0
        self.hopper.thigh_joint.angle = 0.0
        self.hopper.leg_joint.angle = 0.0
        self.hopper.foot_joint.angle = 0.0

class HopperRobot:
    self.top : SlideJoint # indicate the top endpoint of the Hopper
    self.top_joint: HingeJoint # indicate the top hinge joint of the Hopper
    self.thigh_joint : HingeJoint # indicate the hinge joint between Hopper torso and Hopper thigh
    self.leg_joint : HingeJoint # indicate the hinge joint between Hopper thigh and Hopper leg
    self.foot_joint : HingeJoint # indicate the hinge joint between Hopper leg and Hopper foot

class SlideJoint:
    self.position_x : float # x-coordinate position in the world frame, in meters
    self.position_z : float # z-coordinate position in the world frame, in meters
    self.velocity_x : float # x-coordinate velocity in the world frame, in meters per second
    self.velocity_z : float # z-coordinate velocity in the world frame, in meters per second

class HingeJoint:
    self.angle : float # joint angle value of the HingeJoint, in radians
    self.angular_velocity : float # joint angular velocity of the HingeJoint, in radians per second

Here we define that the two-dimensional hopper can only move in the x-coordinate, with positive x-coordinate to be the forward direction and negative x-coordinate to be the backward direction.

I want it to fulfill the following task: {instruction}
1. please think step by step and tell me what does this task mean;
2. then write a function that format as `def compute_dense_reward(self, action) -> float` and returns the `reward : float` only.
3. Take care of variable's type, never use the function of another python class and do not use ungiven variables.
4. When you writing code, you can also add some comments as your thought.
5. Do not give too much weight on regulization or penalty. And give different weight to different reward term, according to its measurement and importance.
""".strip()

HOPPER_PROMPT_CLASS = PromptTemplate(
    input_variables=["instruction"],
    template=classlike_prompt
)