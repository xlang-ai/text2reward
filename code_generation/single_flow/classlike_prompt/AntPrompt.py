from langchain.prompts import PromptTemplate

classlike_prompt = """
You are an expert in robotics locomotion, reinforcement learning and code generation.
We are going to control an Ant to complete given tasks.
The ant is a 3D robot consisting of one torso (free rotational body) with four legs attached to it with each leg having two links. Eight hinges connecting the two links of each leg and the torso (nine parts and eight hinges).
The action space of the Hopper is a Box(-1, 1, (8,), float32).
| Num  | Action                                                            | Joint | Unit         |
| ---- | ----------------------------------------------------------------- | ----- | ------------ |
| 0    | Torque applied on the rotor between the torso and front left hip  | hinge | torque (N m) |
| 1    | Torque applied on the rotor between the front left two links      | hinge | torque (N m) |
| 2    | Torque applied on the rotor between the torso and front right hip | hinge | torque (N m) |
| 3    | Torque applied on the rotor between the front right two links     | hinge | torque (N m) |
| 4    | Torque applied on the rotor between the torso and back left hip   | hinge | torque (N m) |
| 5    | Torque applied on the rotor between the back left two links       | hinge | torque (N m) |
| 6    | Torque applied on the rotor between the torso and back right hip  | hinge | torque (N m) |
| 7    | Torque applied on the rotor between the back right two links      | hinge | torque (N m) |

Now I want you to help me write a reward function of reinforcement learning.
I'll give you the attributes of the environment and Hopper itself. You can use these class attributes to write the reward function.

class BaseEnv(gym.Env):
    self.ant : AntRobot
    self.dt : float # the time between two actions, in seconds, default is 0.05 s
    def __init__(self): # initialzie state of the Ant
        self.ant.torso.position_z = 0.75 # in meters

class AntRobot:
    self.torso : FreeJoint # indicate the torso of the ant
    self.front_left_leg_joint : HingeJoint # indicate the hinge joint between torso and first link on front left
    self.front_left_ankle_joint : HingeJoint # indicate the hinge joint between the two links on the front left
    self.front_right_leg_joint : HingeJoint # indicate the hinge joint between torso and first link on front right
    self.front_right_ankle_joint : HingeJoint # indicate the hinge joint between the two links on the front right
    self.back_left_leg_joint : HingeJoint # indicate the hinge joint between torso and first link on back left
    self.back_left_ankle_joint : HingeJoint # indicate the hinge joint between the two links on the back left
    self.back_right_leg_joint : HingeJoint # indicate the hinge joint between torso and first link on back right
    self.back_right_ankle_joint : HingeJoint # indicate the hinge joint between the two links on the back right

class FreeJoint:
    self.position_x : float # x-coordinate position in the world frame, in meters
    self.position_y : float # y-coordinate position in the world frame, in meters
    self.position_z : float # z-coordinate position in the world frame, in meters
    self.velocity_x : float # x-coordinate velocity in the world frame, in meters per second
    self.velocity_y : float # z-coordinate velocity in the world frame, in meters per second
    self.velocity_z : float # z-coordinate velocity in the world frame, in meters per second
    self.angular_velocity_x : float # x-coordinate angular velocity in the world frame, in radians per second
    self.angular_velocity_y : float # y-coordinate angular velocity in the world frame, in radians per second
    self.angular_velocity_z : float # z-coordinate angular velocity in the world frame, in radians per second

class HingeJoint:
    self.angle : float # joint angle value of the HingeJoint, in radians
    self.angular_velocity : float # joint angular velocity of the HingeJoint, in radians per second

Here we define that positive x-coordinate to be the forward direction and negative x-coordinate to be the backward direction.

I want it to fulfill the following task: {instruction}
1. please think step by step and tell me what does this task mean;
2. then write a function that format as `def compute_dense_reward(self, action) -> float` and returns the `reward : float` only.
3. Take care of variable's type, never use the function of another python class and do not use ungiven variables.
4. When you writing code, you can also add some comments as your thought.
5. Do not give too much weight on regulization or penalty. And give different weight to different reward term, according to its measurement and importance.
""".strip()

ANT_PROMPT_CLASS = PromptTemplate(
    input_variables=["instruction"],
    template=classlike_prompt
)
