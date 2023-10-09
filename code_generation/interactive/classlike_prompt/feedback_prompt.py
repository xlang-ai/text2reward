from langchain.prompts import PromptTemplate

feedback_prompt = """
Generated code shown as below:
```python
{general_code}
```

Feed this reward code into the environment, and use the RL algorithm to train the policy. After training, I can see from the robot that:
{description}

To make the code more accurate and train better robot, the feedback for improvement is:
{feedback}
""".strip()

FEEDBACK_PROMPT = PromptTemplate(
    input_variables=["general_code", "description", "feedback"],
    template=feedback_prompt
)
